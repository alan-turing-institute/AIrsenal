import multiprocessing
import sys

from curl_cffi import requests
from sqlalchemy.orm.session import Session

from airsenal.apply.lineup import set_lineup
from airsenal.apply.transfers import make_transfers
from airsenal.core.concurrency import set_multiprocessing_start_method
from airsenal.core.logging import get_logger
from airsenal.core.season import CURRENT_SEASON, get_past_seasons
from airsenal.db.queries.gameweeks import get_gameweeks_array, next_gameweek
from airsenal.db.queries.tags import get_latest_prediction_tag
from airsenal.db.session import session_scope
from airsenal.export.absences import main as save_expected_absences
from airsenal.fetch.fpl_api import get_fetcher, require_fpl_team_id
from airsenal.ingest.init_db import check_clean_db, make_init_db
from airsenal.ingest.update import update_db
from airsenal.optimization.moves import TransferConstraints
from airsenal.optimization.run_squad import fill_initial_squad
from airsenal.optimization.run_transfers import run_optimization
from airsenal.optimization.transfer_optimizers import (
    TRANSFER_OPTIMIZERS,
    TreeSearchConfig,
)
from airsenal.prediction.protocols import ConfiguredTeamModel, PlayerModel
from airsenal.prediction.registry import (
    DEFAULT_PLAYER_MODEL,
    PLAYER_MODELS,
    build_team_model,
)
from airsenal.prediction.run import make_predictedscore_table
from airsenal.reporting.top_players import get_top_predicted_points
from airsenal.squad.state import get_entry_start_gameweek

logger = get_logger(__name__)


def run_pipeline(
    num_thread: int | None,
    n_gameweeks: int,
    fpl_team_id: int | None,
    clean: bool,
    apply_transfers: bool,
    wildcard_week: int,
    free_hit_week: int,
    triple_captain_week: int,
    bench_boost_week: int,
    n_previous: int,
    no_current_season: bool,
    team_model: str,
    epsilon: float | None,
    max_transfers: int,
    max_hit: int,
    allow_unused: bool,
    save_absences: bool,
    player_model: str = DEFAULT_PLAYER_MODEL,
    player_model_options: dict[str, str] | None = None,
    team_model_options: dict[str, str] | None = None,
) -> None:
    """
    Run the full pipeline, from setting up the database and filling
    with players, teams, fixtures, and results (if it didn't already exist),
    then updating with the latest info, then running predictions to get a
    score estimate for every player, and finally optimization, to choose
    the best squad.
    """
    fpl_team_id = require_fpl_team_id(fpl_team_id)
    logger.info("Running for FPL Team ID %s", fpl_team_id)
    if not num_thread:
        num_thread = multiprocessing.cpu_count()
    set_multiprocessing_start_method()

    gameweeks = get_gameweeks_array(n_gameweeks=n_gameweeks)

    fitted_player_model = PLAYER_MODELS.create_with(
        player_model, player_model_options or {}
    )
    fitted_team_model = build_team_model(team_model, team_model_options, epsilon)

    with session_scope() as dbsession:
        if check_clean_db(clean, dbsession):
            logger.info("[bold]Database Setup[/bold]")
            setup_ok = setup_database(
                fpl_team_id, n_previous, no_current_season, dbsession
            )
            if not setup_ok:
                msg = "Problem setting up initial db"
                raise RuntimeError(msg)
            logger.info("[green]Database setup complete![/green]")
            update_attr = False
        else:
            logger.debug("Found pre-existing AIrsenal database.")
            update_attr = True

        logger.info("[bold]Updating database[/bold]")
        try:
            update_ok = update_database(fpl_team_id, update_attr, dbsession)
        except requests.exceptions.RequestException:
            logger.warning("Database updated failed.", exc_info=True)
            update_ok = False

        if not update_ok:
            confirmed = input(
                "The database update failed. AIrsenal can continue using the latest "
                "status of its database but the results may be outdated or invalid.\n"
                "Do you want to continue? [y/n] "
            )
            if confirmed == "n":
                sys.exit()
        else:
            logger.info("[green]Database update complete![/green]")

        logger.info("[bold]Points Prediction[/bold]")
        predict_ok = run_prediction(
            gameweeks=gameweeks,
            dbsession=dbsession,
            player_model=fitted_player_model,
            team_model=fitted_team_model,
        )
        if not predict_ok:
            msg = "Problem running prediction"
            raise RuntimeError(msg)
        logger.info("[green]Prediction complete![/green]")

        chips_played = setup_chips(
            wildcard_week=wildcard_week,
            free_hit_week=free_hit_week,
            triple_captain_week=triple_captain_week,
            bench_boost_week=bench_boost_week,
        )
        if get_entry_start_gameweek(fpl_team_id, get_fetcher()) == next_gameweek():
            logger.info("[bold]Generating Squad[/bold]")
            new_squad_ok = run_make_squad(
                gameweeks,
                fpl_team_id,
                dbsession,
                chip_gameweeks=chips_played,
            )
            if not new_squad_ok:
                msg = "Problem creating a new squad"
                raise RuntimeError(msg)
        else:
            logger.info("[bold]Optimising Transfers[/bold]")
            opt_ok = run_optimize_squad(
                num_thread=num_thread,
                gameweeks=gameweeks,
                fpl_team_id=fpl_team_id,
                dbsession=dbsession,
                chips_played=chips_played,
                max_transfers=max_transfers,
                max_hit=max_hit,
                allow_unused=allow_unused,
            )
            if not opt_ok:
                msg = "Problem running optimization"
                raise RuntimeError(msg)

        logger.info("[green]Optimization complete![/green]")
        if apply_transfers:
            logger.info("[bold]Applying Transfers[/bold]")
            transfers_ok = make_transfers(fpl_team_id)
            if not transfers_ok:
                msg = "Problem applying the transfers"
                raise RuntimeError(msg)
            logger.info("[bold]Setting Lineup[/bold]")
            lineup_ok = set_starting_11(fpl_team_id)
            if not lineup_ok:
                msg = "Problem setting the lineup"
                raise RuntimeError(msg)
        if save_absences:
            logger.info("[bold]Saving Absences[/bold]")
            save_expected_absences()
        logger.info("[green]Pipeline finished![/green]")


def setup_database(
    fpl_team_id: int, n_previous: int, no_current_season: bool, dbsession: Session
) -> bool:
    """
    Set up database
    """
    if no_current_season:
        seasons = get_past_seasons(n_previous)
    else:
        seasons = [CURRENT_SEASON, *get_past_seasons(n_previous)]

    return make_init_db(fpl_team_id, seasons, dbsession)


def setup_chips(
    wildcard_week: int,
    free_hit_week: int,
    triple_captain_week: int,
    bench_boost_week: int,
) -> dict[str, int]:
    """
    Set up chips to be played for particular gameweeks. Specifically: wildcard,
    free_hit, triple_captain, bench_boost
    """
    return {
        "wildcard": wildcard_week,
        "free_hit": free_hit_week,
        "triple_captain": triple_captain_week,
        "bench_boost": bench_boost_week,
    }


def update_database(fpl_team_id: int, attr: bool, dbsession: Session) -> bool:
    """
    Update database
    """
    season = CURRENT_SEASON
    return update_db(season, attr, fpl_team_id, dbsession)


def run_prediction(
    gameweeks: list[int],
    dbsession: Session,
    player_model: PlayerModel | None = None,
    team_model: ConfiguredTeamModel | None = None,
) -> bool:
    """
    Run prediction
    """
    season = CURRENT_SEASON
    tag = make_predictedscore_table(
        gameweeks=gameweeks,
        season=season,
        include_bonus=True,
        include_cards=True,
        include_saves=True,
        player_model=player_model,
        team_model=team_model,
        dbsession=dbsession,
    )

    # print players with top predicted points
    get_top_predicted_points(
        gameweeks=gameweeks,
        tag=tag,
        season=season,
        per_position=True,
        n_players=5,
        dbsession=dbsession,
    )
    return True


def run_make_squad(
    gameweeks: list[int],
    fpl_team_id: int,
    dbsession: Session,
    chip_gameweeks: dict[str, int] | None = None,
) -> bool:
    """Build the initial squad."""
    season = CURRENT_SEASON
    tag = get_latest_prediction_tag(season, tag_prefix="", dbsession=dbsession)

    fill_initial_squad(
        tag,
        gameweeks,
        season,
        fpl_team_id,
        chip_gameweeks=chip_gameweeks,
    )

    return True


def run_optimize_squad(
    num_thread: int,
    gameweeks: list[int],
    fpl_team_id: int,
    dbsession: Session,
    chips_played: dict[str, int],
    max_transfers: int,
    max_hit: int,
    allow_unused: bool,
) -> bool:
    """
    Build the initial squad
    """
    season = CURRENT_SEASON
    tag = get_latest_prediction_tag(season, tag_prefix="", dbsession=dbsession)
    run_optimization(
        gameweeks=gameweeks,
        tag=tag,
        season=season,
        fpl_team_id=fpl_team_id,
        chip_gameweeks=chips_played,
        constraints=TransferConstraints(
            max_total_hit=max_hit,
            allow_unused_transfers=allow_unused,
            max_opt_transfers=max_transfers,
        ),
        optimizer=TRANSFER_OPTIMIZERS.create(
            "tree_search", TreeSearchConfig(num_thread=num_thread)
        ),
    )
    return True


def set_starting_11(fpl_team_id: int | None = None) -> bool:
    """
    Set the lineup based on the latest optimization run.

    """
    set_lineup(fpl_team_id)
    return True
