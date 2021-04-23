import sys
import os
import multiprocessing

import click

from airsenal.framework.db_config import AIrsenalDBFile
from airsenal.framework.schema import session, Team
from airsenal.framework.utils import NEXT_GAMEWEEK, fetcher


@click.command("airsenal_run_pipeline")
@click.option(
    "--num_thread",
    type=int,
    help="No. of threads to use for pipeline run",
)
@click.option(
    "--num_iterations",
    type=int,
    default=10,
    help="No. of iterations for generating initial squad (start of season only)",
)
@click.option(
    "--weeks_ahead", type=int, default=3, help="No of weeks to use for pipeline run"
)
@click.option(
    "--num_free_transfers",
    type=int,
    default=1,
    help="Number of free transfer for pipeline run",
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
def run_pipeline(
    num_thread, num_iterations, weeks_ahead, num_free_transfers, fpl_team_id, clean
):
    if fpl_team_id is None:
        fpl_team_id = fetcher.FPL_TEAM_ID
    print("Running for FPL Team ID {}".format(fpl_team_id))
    if not num_thread:
        num_thread = multiprocessing.cpu_count()
    if clean:
        click.echo("Cleaning database..")
        clean_database()
    if database_is_empty():
        click.echo("Setting up Database..")
        setup_database(fpl_team_id)
        click.echo("Database setup complete..")
        update_attr = False
    else:
        click.echo("Found pre-existing AIrsenal database.")
        update_attr = True
    click.echo("Updating database..")
    update_database(fpl_team_id, attr=update_attr)
    click.echo("Database update complete..")
    click.echo("Running prediction..")
    run_prediction(num_thread, weeks_ahead)
    click.echo("Prediction complete..")
    if NEXT_GAMEWEEK == 1:
        click.echo("Generating a squad..")
        run_make_team(num_iterations, weeks_ahead)
        click.echo("Optimization complete..")
    else:
        click.echo("Running optimization..")
        run_optimization(num_thread, weeks_ahead, num_free_transfers, fpl_team_id)
        click.echo("Optimization complete..")
    click.echo("Applying suggested transfers...")
    make_transfers(fpl_team_id)
    


def clean_database():
    """
    Clean up database
    """
    try:
        if os.path.exists(AIrsenalDBFile):
            os.remove(AIrsenalDBFile)
    except IOError as exc:
        click.echo(
            "Error while deleting file {}. Reason:{}".format(AIrsenalDBFile, exc)
        )
        sys.exit(1)


def database_is_empty():
    """
    Basic check to determine whether the database is empty
    """
    return session.query(Team).first() is None


def setup_database(fpl_team_id):
    """
    Set up database
    """
    os.system("airsenal_setup_initial_db --fpl_team_id {}".format(fpl_team_id))


def update_database(fpl_team_id, attr=True):
    """
    Update database
    """
    if attr:
        os.system("airsenal_update_db --fpl_team_id {}".format(fpl_team_id))
    else:
        os.system("airsenal_update_db --noattr --fpl_team_id {}".format(fpl_team_id))


def run_prediction(num_thread, weeks_ahead):
    """
    Run prediction
    """
    cmd = "airsenal_run_prediction --num_thread {} --weeks_ahead {}".format(
        num_thread, weeks_ahead
    )
    os.system(cmd)


def run_make_team(num_iterations, weeks_ahead):
    """
    Run optimization
    """
    cmd = "airsenal_make_squad --num_iterations {} --num_gw {}".format(
        num_iterations, weeks_ahead
    )
    os.system(cmd)


def run_optimization(num_thread, weeks_ahead, num_free_transfers, fpl_team_id):
    """
    Run optimization
    """
    cmd = (
        "airsenal_run_optimization --num_thread {} --weeks_ahead {}  "
        "--num_free_transfers {} --fpl_team_id {}"
    ).format(num_thread, weeks_ahead, num_free_transfers, fpl_team_id)
    os.system(cmd)

def make_transfers(fpl_team_id = None):
    """
    Post transfers from transfer suggestion table.

    Team id not necessary as will be taken from transfer suggestion table.
    """

    cmd = (
        "airsenal_make_transfers --fpl_team_id {}"
    ).format(fpl_team_id)
    os.system(cmd)


def main():
    sys.exit()


if __name__ == "__main__":
    main()
