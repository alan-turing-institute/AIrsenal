import multiprocessing
import sys
import warnings

import click
from tqdm import TqdmWarning

from airsenal.framework.multiprocessing_utils import set_multiprocessing_start_method
from airsenal.framework.optimization_pygmo import make_new_squad_pygmo
from airsenal.framework.optimization_utils import fill_initial_suggestion_table
from airsenal.framework.schema import session_scope
from airsenal.framework.utils import (
    CURRENT_SEASON,
    NEXT_GAMEWEEK,
    fetcher,
    get_latest_prediction_tag,
    get_entry_start_gameweek,
)
from airsenal.scripts.fill_db_init import check_clean_db, make_init_db
from airsenal.scripts.fill_predictedscore_table import (
    get_top_predicted_points,
    make_predictedscore_table,
)
from airsenal.scripts.fill_transfersuggestion_table import run_optimization
from airsenal.scripts.make_transfers import make_transfers
from airsenal.scripts.set_lineup import set_lineup
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
)
@click.option(
    "--free_hit_week",
    type=int,
    help="Play free hit in the specified week. Choose 0 for 'any week'.",
)
@click.option(
    "--triple_captain_week",
    type=int,
    help="Play triple captain in the specified week. Choose 0 for 'any week'.",
)
@click.option(
    "--bench_boost_week",
    type=int,
    help="Play bench_boost in the specified week. Choose 0 for 'any week'.",
)
def run_pipeline(
    num_thread,
    weeks_ahead,
    fpl_team_id,
    clean,
    apply_transfers,
    wildcard_week,
    free_hit_week,
    triple_captain_week,
    bench_boost_week,
):
    """
    Run the full pipeline, from setting up the database and filling
    with players, teams, fixtures, and results (if it didn't already exist),
    then updating with the latest info, then running predictions to get a
    score estimate for every player, and finally optimization, to choose
    the best squad.
    """
    if fpl_team_id is None:
        fpl_team_id = fetcher.FPL_TEAM_ID
    print("Running for FPL Team ID {}".format(fpl_team_id))
    if not num_thread:
        num_thread = multiprocessing.cpu_count()
    set_multiprocessing_start_method(num_thread)

    with session_scope() as dbsession:
        if check_clean_db(clean, dbsession):
            click.echo("Setting up Database..")
            setup_ok = setup_database(fpl_team_id, dbsession)
            if not setup_ok:
                raise RuntimeError("Problem setting up initial db")
            click.echo("Database setup complete..")
            update_attr = False
        else:
            click.echo("Found pre-existing AIrsenal database.")
            update_attr = True
        click.echo("Updating database..")
        update_ok = update_database(fpl_team_id, update_attr, dbsession)
        if not update_ok:
            raise RuntimeError("Problem updating db")
        click.echo("Database update complete..")
        click.echo("Running prediction..")
        predict_ok = run_prediction(num_thread, weeks_ahead, dbsession)
        if not predict_ok:
            raise RuntimeError("Problem running prediction")
        click.echo("Prediction complete..")
        if NEXT_GAMEWEEK == get_entry_start_gameweek(fpl_team_id, fetcher):
            click.echo("Generating a squad..")
            new_squad_ok = run_make_squad(weeks_ahead, fpl_team_id, dbsession)
            if not new_squad_ok:
                raise RuntimeError("Problem creating a new squad")
        else:
            click.echo("Running optimization..")
            chips_played = setup_chips(
                wildcard_week, free_hit_week, triple_captain_week, bench_boost_week
            )
            opt_ok = run_optimize_squad(
                num_thread, weeks_ahead, fpl_team_id, dbsession, chips_played
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


def setup_database(fpl_team_id, dbsession):
    """
    Set up database
    """
    return make_init_db(fpl_team_id, dbsession)


def setup_chips(wildcard_week, free_hit_week, triple_captain_week, bench_boost_week):
    """
    Set up chips to be played for particular gameweeks. Specifically: wildcard,
    free_hit, triple_captain, bench_boost
    """
    chips_gameweeks = {}
    if wildcard_week:
        chips_gameweeks["wildcard"] = wildcard_week
    if free_hit_week:
        chips_gameweeks["free_hit"] = free_hit_week
    if triple_captain_week:
        chips_gameweeks["triple_captain"] = triple_captain_week
    if bench_boost_week:
        chips_gameweeks["bench_boost"] = bench_boost_week

    return chips_gameweeks


def update_database(fpl_team_id, attr, dbsession):
    """
    Update database
    """
    season = CURRENT_SEASON
    return update_db(season, attr, fpl_team_id, dbsession)


def run_prediction(num_thread, weeks_ahead, dbsession):
    """
    Run prediction
    """
    gw_range = list(range(NEXT_GAMEWEEK, NEXT_GAMEWEEK + weeks_ahead))
    season = CURRENT_SEASON
    tag = make_predictedscore_table(
        gw_range=gw_range,
        season=season,
        num_thread=num_thread,
        include_bonus=True,
        include_cards=True,
        include_saves=True,
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


def run_make_squad(weeks_ahead, fpl_team_id, dbsession):
    """
    Build the initial squad
    """
    gw_range = list(range(NEXT_GAMEWEEK, NEXT_GAMEWEEK + weeks_ahead))
    season = CURRENT_SEASON
    tag = get_latest_prediction_tag(season, tag_prefix="", dbsession=dbsession)

    best_squad = make_new_squad_pygmo(
        gw_range,
        tag,
    )
    best_squad.get_expected_points(NEXT_GAMEWEEK, tag)
    print(best_squad)
    fill_initial_suggestion_table(
        best_squad, fpl_team_id, tag, season, NEXT_GAMEWEEK, dbsession=dbsession
    )
    return True


def run_optimize_squad(num_thread, weeks_ahead, fpl_team_id, dbsession, chips_played):
    """
    Build the initial squad
    """
    gw_range = list(range(NEXT_GAMEWEEK, NEXT_GAMEWEEK + weeks_ahead))
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
        )
    return True


def set_starting_11(fpl_team_id=None):
    """
    Set the lineup based on the latest optimization run.

    """
    set_lineup(fpl_team_id)
    return True


def main():
    sys.exit()


if __name__ == "__main__":
    main()
