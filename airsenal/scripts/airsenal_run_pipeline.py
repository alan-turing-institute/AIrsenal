import sys
import os
import multiprocessing
import warnings
import click
from tqdm import TqdmWarning

from airsenal.framework.db_config import AIrsenalDBFile
from airsenal.framework.schema import session_scope, Team, Base
from airsenal.framework.utils import (
    CURRENT_SEASON,
    NEXT_GAMEWEEK,
    fetcher,
    get_latest_prediction_tag,
)
from airsenal.framework.optimization_utils import fill_initial_suggestion_table
from airsenal.framework.optimization_pygmo import make_new_squad
from airsenal.scripts.fill_db_init import make_init_db
from airsenal.scripts.update_db import update_db
from airsenal.scripts.fill_predictedscore_table import (
    make_predictedscore_table,
    get_top_predicted_points,
)
from airsenal.scripts.fill_transfersuggestion_table import run_optimization
from airsenal.scripts.make_transfers import make_transfers
from airsenal.scripts.set_lineup import set_lineup

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
def run_pipeline(num_thread, weeks_ahead, fpl_team_id, clean, apply_transfers):
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

    with session_scope() as dbsession:
        if clean:
            click.echo("Cleaning database..")
            clean_database()
        if database_is_empty(dbsession):
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
        if NEXT_GAMEWEEK == 1:
            click.echo("Generating a squad..")
            new_squad_ok = run_make_squad(weeks_ahead, fpl_team_id, dbsession)
            if not new_squad_ok:
                raise RuntimeError("Problem creating a new squad")
        else:
            click.echo("Running optimization..")
            opt_ok = run_optimize_squad(num_thread, weeks_ahead, fpl_team_id, dbsession)
            if not opt_ok:
                raise RuntimeError("Problem running optimization")

        click.echo("Optimization complete..")
        if apply_transfers:
            click.echo("Applying suggested transfers...")
            transfers_ok = make_transfers(fpl_team_id)
            if not transfers_ok:
                raise RuntimeError("Problem applying the transfers")
            click.echo("Setting Lineup...")
            lineup_ok = set_lineup(fpl_team_id)
            if not lineup_ok:
                raise RuntimeError("Problem setting the lineup")
        click.echo("Pipeline finished OK!")


def clean_database():
    """
    Clean up database
    """
    Base.metadata.drop_all()
    Base.metadata.create_all()


def database_is_empty(dbsession):
    """
    Basic check to determine whether the database is empty
    """
    if os.path.exists(AIrsenalDBFile):
        return dbsession.query(Team).first() is None
    else:  # file doesn't exist - db is definitely empty!
        return True


def setup_database(fpl_team_id, dbsession):
    """
    Set up database
    """
    return make_init_db(fpl_team_id, dbsession)


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

    best_squad = make_new_squad(
        gw_range,
        tag,
    )
    best_squad.get_expected_points(NEXT_GAMEWEEK, tag)
    print(best_squad)
    fill_initial_suggestion_table(
        best_squad, fpl_team_id, tag, season, NEXT_GAMEWEEK, dbsession=dbsession
    )
    return True


def run_optimize_squad(num_thread, weeks_ahead, fpl_team_id, dbsession):
    """
    Run optimization
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
        )
    return True


def set_lineup(fpl_team_id=None):
    """
    Set the lineup based on the latest optimization run.

    """
    set_lineup(fpl_team_id)
    return True


def main():
    sys.exit()


if __name__ == "__main__":
    main()
