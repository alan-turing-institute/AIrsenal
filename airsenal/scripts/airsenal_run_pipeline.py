import multiprocessing
import sys
import warnings
from typing import List, Optional, Union

import click
import requests
from bpl import ExtendedDixonColesMatchPredictor, NeutralDixonColesMatchPredictor
from sqlalchemy.orm.session import Session
from tqdm import TqdmWarning

from airsenal.framework.multiprocessing_utils import set_multiprocessing_start_method
from airsenal.framework.schema import session_scope
from airsenal.framework.utils import (
    CURRENT_SEASON,
    NEXT_GAMEWEEK,
    fetcher,
    get_entry_start_gameweek,
    get_gameweeks_array,
    get_latest_prediction_tag,
    get_past_seasons,
    parse_team_model_from_str,
)
from airsenal.scripts.fill_db_init import check_clean_db, make_init_db
from airsenal.scripts.fill_predictedscore_table import (
    get_top_predicted_points,
    make_predictedscore_table,
)
from airsenal.scripts.fill_transfersuggestion_table import run_optimization
from airsenal.scripts.make_transfers import make_transfers
from airsenal.scripts.set_lineup import set_lineup
from airsenal.scripts.squad_builder import fill_initial_squad
from airsenal.scripts.update_db import update_db


@click.command("airsenal_run_pipeline")
@click.option(
    "--num_thread",
    type=int,
    help="No. of threads to use for pipeline run",
)
@click.option(
    "--weeks_ahead", type=int, default=3, help="No of weeks to use for pipeline run"
)
@click.option(
    "--fpl_team_id",
    type=int,
    required=False,
    help="fpl team id for pipeline run",
)
@click.option(
    "--clean",
    is_flag=True,
    help="If set, delete and recreate the AIrsenal database",
)
@click.option(
    "--apply_transfers",
    is_flag=True,
    help="If set, go ahead and make the transfers via the API.",
)
@click.option(
    "--wildcard_week",
    type=int,
    help=(
        "If set to 0, consider playing wildcard in any gameweek. "
        "If set to a specific gameweek, it'll be played for that particular gameweek."
    ),
    default=-1,
)
@click.option(
    "--free_hit_week",
    type=int,
    help="Play free hit in the specified week. Choose 0 for 'any week'.",
    default=-1,
)
@click.option(
    "--triple_captain_week",
    type=int,
    help="Play triple captain in the specified week. Choose 0 for 'any week'.",
    default=-1,
)
@click.option(
    "--bench_boost_week",
    type=int,
    help="Play bench_boost in the specified week. Choose 0 for 'any week'.",
    default=-1,
)
@click.option(
    "--n_previous",
    help="specify how many seasons to look back into the past for (defaults to 3)",
    type=int,
    default=3,
)
@click.option(
    "--no_current_season",
    help="If set, does not include CURRENT_SEASON in database",
    is_flag=True,
)
@click.option(
    "--team_model",
    help="which team model to fit",
    type=click.Choice(["extended", "neutral"]),
    default="extended",
)
@click.option(
    "--epsilon",
    help="how much to downweight games by in exponential time weighting",
    type=float,
    default=0.0,
)
@click.option(
    "--max_transfers",
    help="specify maximum number of transfers to be made each gameweek (defaults to 2)",
    type=click.IntRange(min=0, max=2),
    default=2,
)
@click.option(
    "--max_hit",
    help=(
        "specify maximum number of points to spend on additional transfers "
        "(defaults to 8)"
    ),
    type=click.IntRange(min=0),
    default=8,
)
@click.option(
    "--allow_unused",
    help="If set, include strategies that waste free transfers",
    is_flag=True,
)
def run_pipeline(
    num_thread: int,
    weeks_ahead: int,
    fpl_team_id: int,
    clean: bool,
    apply_transfers: bool,
    wildcard_week: int,
    free_hit_week: int,
    triple_captain_week: int,
    bench_boost_week: int,
    n_previous: int,
    no_current_season: bool,
    team_model: str,
    epsilon: int,
    max_transfers: int,
    max_hit: int,
    allow_unused: bool,
) -> None:
    """
    Run the full pipeline, from setting up the database and filling
    with players, teams, fixtures, and results (if it didn't already exist),
    then updating with the latest info, then running predictions to get a
    score estimate for every player, and finally optimization, to choose
    the best squad.
    """
    if fpl_team_id is None:
        fpl_team_id = fetcher.FPL_TEAM_ID
    print(f"Running for FPL Team ID {fpl_team_id}")
    if not num_thread:
        num_thread = multiprocessing.cpu_count()
    set_multiprocessing_start_method()

    gw_range = get_gameweeks_array(weeks_ahead=weeks_ahead)

    team_model_class = parse_team_model_from_str(team_model)

    with session_scope() as dbsession:
        if check_clean_db(clean, dbsession):
            click.echo("Setting up Database..")
            setup_ok = setup_database(
                fpl_team_id, n_previous, no_current_season, dbsession
            )
            if not setup_ok:
                raise RuntimeError("Problem setting up initial db")
            click.echo("Database setup complete..")
            update_attr = False
        else:
            click.echo("Found pre-existing AIrsenal database.")
            update_attr = True

        click.echo("Updating database..")
        try:
            update_ok = update_database(fpl_team_id, update_attr, dbsession)
        except requests.exceptions.RequestException as e:
            warnings.warn(f"Database updated failed: {e}")
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
            click.echo("Database update complete..")

        click.echo("Running prediction..")
        predict_ok = run_prediction(
            num_thread=num_thread,
            gw_range=gw_range,
            dbsession=dbsession,
            team_model=team_model_class,
            team_model_args={"epsilon": epsilon},
        )
        if not predict_ok:
            raise RuntimeError("Problem running prediction")
        click.echo("Prediction complete..")

        if NEXT_GAMEWEEK == get_entry_start_gameweek(fpl_team_id, fetcher):
            click.echo("Generating a squad..")
            new_squad_ok = run_make_squad(gw_range, fpl_team_id, dbsession)
            if not new_squad_ok:
                raise RuntimeError("Problem creating a new squad")
        else:
            click.echo("Running optimization..")
            chips_played = setup_chips(
                wildcard_week=wildcard_week,
                free_hit_week=free_hit_week,
                triple_captain_week=triple_captain_week,
                bench_boost_week=bench_boost_week,
            )
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
                raise RuntimeError("Problem running optimization")

        click.echo("Optimization complete..")
        if apply_transfers:
            click.echo("Applying suggested transfers...")
            transfers_ok = make_transfers(fpl_team_id)
            if not transfers_ok:
                raise RuntimeError("Problem applying the transfers")
            click.echo("Setting Lineup...")
            lineup_ok = set_starting_11(fpl_team_id)
            if not lineup_ok:
                raise RuntimeError("Problem setting the lineup")

        click.echo("Pipeline finished OK!")


def setup_database(
    fpl_team_id: int, n_previous: int, no_current_season: bool, dbsession: Session
) -> bool:
    """
    Set up database
    """
    if no_current_season:
        seasons = get_past_seasons(n_previous)
    else:
        seasons = [CURRENT_SEASON] + get_past_seasons(n_previous)

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
    num_thread: int,
    gw_range: List[int],
    dbsession: Session,
    team_model: Union[
        ExtendedDixonColesMatchPredictor, NeutralDixonColesMatchPredictor
    ] = ExtendedDixonColesMatchPredictor(),
    team_model_args: dict = {"epsilon": 0.0},
) -> bool:
    """
    Run prediction
    """
    season = CURRENT_SEASON
    tag = make_predictedscore_table(
        gw_range=gw_range,
        season=season,
        num_thread=num_thread,
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


def run_make_squad(gw_range: List[int], fpl_team_id: int, dbsession: Session) -> bool:
    """
    Build the initial squad
    """
    season = CURRENT_SEASON
    tag = get_latest_prediction_tag(season, tag_prefix="", dbsession=dbsession)

    fill_initial_squad(tag, gw_range, season, fpl_team_id)

    return True


def run_optimize_squad(
    num_thread: int,
    gw_range: List[int],
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", TqdmWarning)
        run_optimization(
            gameweeks=gw_range,
            tag=tag,
            season=season,
            fpl_team_id=fpl_team_id,
            num_thread=num_thread,
            chip_gameweeks=chips_played,
            max_transfers=max_transfers,
            max_total_hit=max_hit,
            allow_unused_transfers=allow_unused,
        )
    return True


def set_starting_11(fpl_team_id: Optional[int] = None) -> bool:
    """
    Set the lineup based on the latest optimization run.

    """
    set_lineup(fpl_team_id)
    return True


def main():
    sys.exit()


if __name__ == "__main__":
    main()
