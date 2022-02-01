import click

from airsenal.framework.schema import session_scope
from airsenal.framework.utils import CURRENT_SEASON
from airsenal.scripts.dump_db_contents import dump_database
from airsenal.scripts.fill_db_init import check_clean_db, make_init_db
from airsenal.scripts.update_db import update_db


@click.command()
@click.option(
    "--setup", is_flag=True, help="Set up the initial database for `airsenal`."
)
@click.option(
    "--update", is_flag=True, help="Update the database with the most recent data."
)
@click.option(
    "--fpl-team-id", type=int, required=False, default=None, help="FPL team ID"
)
@click.option(
    "--dump", is_flag=True, help="Dump database to CSV files in `airsenal/data`."
)
@click.option(
    "--rebuild",
    is_flag=True,
    help="If set, deletes and recreates any pre-existing AIrsenal database.",
)
@click.option(
    "--noattr", is_flag=True, help="Set if you don't want to update player attributes."
)
@click.option("--season", default=CURRENT_SEASON)
def database(setup, update, dump, fpl_team_id, rebuild, noattr, season):
    """
    Setup/update AIrsenal database.
    """
    if setup:
        click.echo("Setting up database")
        with session_scope() as dbsession:
            continue_setup = check_clean_db(rebuild, dbsession)
            if continue_setup:
                make_init_db(fpl_team_id, dbsession)
        # TODO: Add database sanity check to setup once it's fixed
    if update:
        click.echo(f"Updating database for Team ID {fpl_team_id}")
        with session_scope() as dbsession:
            update_db(season, not noattr, fpl_team_id, dbsession)

    if dump:
        dump_database()

    return
