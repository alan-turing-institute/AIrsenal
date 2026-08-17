import multiprocessing
import sys
import warnings

from bpl import ExtendedDixonColesMatchPredictor, NeutralDixonColesMatchPredictor
from curl_cffi import requests
from sqlalchemy.orm.session import Session

from airsenal.framework.bpl_interface import (
    DEFAULT_TEAM_EPSILON,
    parse_team_model_from_str,
)
from airsenal.framework.multiprocessing_utils import set_multiprocessing_start_method
from airsenal.framework.output import print
from airsenal.framework.random_team_model import RandomMatchPredictor
from airsenal.framework.schema import session_scope
from airsenal.framework.utils import (
    CURRENT_SEASON,
    NEXT_GAMEWEEK,
    fetcher,
    get_entry_start_gameweek,
    get_gameweeks_array,
    get_latest_prediction_tag,
    get_past_seasons,
)
from airsenal.scripts.fill_db_init import check_clean_db, make_init_db
from airsenal.scripts.fill_predictedscore_table import (
    get_top_predicted_points,
    make_predictedscore_table,
)
from airsenal.scripts.fill_transfersuggestion_table import run_optimization
from airsenal.scripts.make_transfers import make_transfers
from airsenal.scripts.save_expected_absences import main as save_expected_absences
from airsenal.scripts.set_lineup import set_lineup
from airsenal.scripts.squad_builder import fill_initial_squad
from airsenal.scripts.update_db import update_db


def run_pipeline(
    num_thread: int | None,
    weeks_ahead: int,
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
    epsilon: float,
    max_transfers: int,
    max_hit: int,
    allow_unused: bool,
    save_absences: bool,
) -> None:
    """
    Run the full pipeline, from setting up the database and filling
    with players, teams, fixtures, and results (if it didn't already exist),
    then updating with the latest info, then running predictions to get a
    score estimate for every player, and finally optimization, to choose
    the best squad.
    """
    if fpl_team_id is None:
        if not fetcher.FPL_TEAM_ID:
            msg = "FPL Team ID not provided and not found in environment variables."
            raise RuntimeError(msg)
        fpl_team_id = fetcher.FPL_TEAM_ID
    print(f"Running for FPL Team ID {fpl_team_id}")
    if not num_thread:
        num_thread = multiprocessing.cpu_count()
    set_multiprocessing_start_method()

    gw_range = get_gameweeks_array(weeks_ahead=weeks_ahead)

    team_model_class = parse_team_model_from_str(team_model)

    with session_scope() as dbsession:
        if check_clean_db(clean, dbsession):
            print("Setting up Database..")
            setup_ok = setup_database(
                fpl_team_id, n_previous, no_current_season, dbsession
            )
            if not setup_ok:
                msg = "Problem setting up initial db"
                raise RuntimeError(msg)
            print("Database setup complete..")
            update_attr = False
        else:
            print("Found pre-existing AIrsenal database.")
            update_attr = True

        print("Updating database..")
        try:
            update_ok = update_database(fpl_team_id, update_attr, dbsession)
        except requests.exceptions.RequestException as e:
            warnings.warn(f"Database updated failed: {e}", stacklevel=2)
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
            print("Database update complete..")

        print("Running prediction..")
        predict_ok = run_prediction(
            gw_range=gw_range,
            dbsession=dbsession,
            team_model=team_model_class,
            team_model_args={"epsilon": epsilon},
        )
        if not predict_ok:
            msg = "Problem running prediction"
            raise RuntimeError(msg)
        print("Prediction complete..")

        chips_played = setup_chips(
            wildcard_week=wildcard_week,
            free_hit_week=free_hit_week,
            triple_captain_week=triple_captain_week,
            bench_boost_week=bench_boost_week,
        )
        if get_entry_start_gameweek(fpl_team_id, fetcher) == NEXT_GAMEWEEK:
            print("Generating a squad..")
            new_squad_ok = run_make_squad(
                gw_range,
                fpl_team_id,
                dbsession,
                chip_gameweeks=chips_played,
            )
            if not new_squad_ok:
                msg = "Problem creating a new squad"
                raise RuntimeError(msg)
        else:
            print("Running optimization..")
            opt_ok = run_optimize_squad(
                num_thread=num_thread,
                gw_range=gw_range,
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

        print("Optimization complete..")
        if apply_transfers:
            print("Applying suggested transfers...")
            transfers_ok = make_transfers(fpl_team_id)
            if not transfers_ok:
                msg = "Problem applying the transfers"
                raise RuntimeError(msg)
            print("Setting Lineup...")
            lineup_ok = set_starting_11(fpl_team_id)
            if not lineup_ok:
                msg = "Problem setting the lineup"
                raise RuntimeError(msg)
        if save_absences:
            print("Saving absences to csv...")
            save_expected_absences()
        print("Pipeline finished OK!")


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
) -> dict:
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
    gw_range: list[int],
    dbsession: Session,
    team_model: ExtendedDixonColesMatchPredictor
    | NeutralDixonColesMatchPredictor
    | RandomMatchPredictor
    | None = None,
    team_model_args: dict | None = None,
) -> bool:
    """
    Run prediction
    """
    if team_model_args is None:
        team_model_args = {"epsilon": DEFAULT_TEAM_EPSILON}
    season = CURRENT_SEASON
    tag = make_predictedscore_table(
        gw_range=gw_range,
        season=season,
        include_bonus=True,
        include_cards=True,
        include_saves=True,
        team_model=team_model,
        team_model_args=team_model_args,
        dbsession=dbsession,
    )

    # print players with top predicted points
    get_top_predicted_points(
        gameweek=gw_range,
        tag=tag,
        season=season,
        per_position=True,
        n_players=5,
        dbsession=dbsession,
    )
    return True


def run_make_squad(
    gw_range: list[int],
    fpl_team_id: int,
    dbsession: Session,
    chip_gameweeks: dict[str, int] | None = None,
) -> bool:
    """Build the initial squad."""
    season = CURRENT_SEASON
    tag = get_latest_prediction_tag(season, tag_prefix="", dbsession=dbsession)

    fill_initial_squad(
        tag,
        gw_range,
        season,
        fpl_team_id,
        chip_gameweeks=chip_gameweeks,
    )

    return True


def run_optimize_squad(
    num_thread: int,
    gw_range: list[int],
    fpl_team_id: int,
    dbsession: Session,
    chips_played: dict,
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
        gameweeks=gw_range,
        tag=tag,
        season=season,
        fpl_team_id=fpl_team_id,
        num_thread=num_thread,
        chip_gameweeks=chips_played,
        max_opt_transfers=max_transfers,
        max_total_hit=max_hit,
        allow_unused_transfers=allow_unused,
    )
    return True


def set_starting_11(fpl_team_id: int | None = None) -> bool:
    """
    Set the lineup based on the latest optimization run.

    """
    set_lineup(fpl_team_id)
    return True
