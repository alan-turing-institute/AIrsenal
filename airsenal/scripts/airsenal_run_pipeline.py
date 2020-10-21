import sys
import os
import multiprocessing

import click

from airsenal import TMPDIR
from airsenal.framework.utils import NEXT_GAMEWEEK


@click.command("airsenal_run_pipeline")
@click.option(
    "--num_thread",
    type=int,
    default=None,
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
def airsenal_run_pipeline(
    num_thread, num_iterations, weeks_ahead, num_free_transfers
):
    if not num_thread:
        num_thread = multiprocessing.cpu_count()
    click.echo("Cleaning database..")
    clean_database()
    click.echo("Setting up Database..")
    setup_database()
    click.echo("Database setup complete..")
    click.echo("Updating database..")
    update_database()
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
        run_optimization(num_thread, weeks_ahead, num_free_transfers)
        click.echo("Optimization complete..")


def clean_database():
    """
    Clean up database
    """
    file_path = "{}/data.db".format(TMPDIR)
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except IOError as exc:
        click.echo("Error while deleting file {}. Reason:{}".format(file_path, exc))
        sys.exit(1)


def setup_database():
    """
    Set up database
    """
    os.system("airsenal_setup_initial_db")


def update_database():
    """
    Update database
    """
    os.system("airsenal_update_db --noattr")


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


def run_optimization(num_thread, weeks_ahead, num_free_transfers):
    """
    Run optimization
    """
    cmd = "airsenal_run_optimization --num_thread {} --weeks_ahead {} --num_free_transfers {}".format(
        num_thread, weeks_ahead, num_free_transfers
    )
    os.system(cmd)


def main():
    sys.exit()


if __name__ == "__main__":
    main()
