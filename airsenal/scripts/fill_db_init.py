"""Script to fill the database after install."""
from airsenal.framework.schema import clean_database, database_is_empty
from airsenal.scripts.fill_team_table import make_team_table
from airsenal.scripts.fill_player_table import make_player_table
from airsenal.scripts.fill_player_attributes_table import make_attributes_table
from airsenal.scripts.fill_fixture_table import make_fixture_table
from airsenal.scripts.fill_result_table import make_result_table
from airsenal.scripts.fill_playerscore_table import make_playerscore_table
from airsenal.scripts.fill_fifa_ratings_table import make_fifa_ratings_table

from airsenal.framework.transaction_utils import fill_initial_squad
from airsenal.framework.schema import session_scope

import argparse


def check_clean_db(clean, dbsession):
    """Check whether an AIrsenal database already exists. If clean is True attempt to
    delete any pre-existing database first. Returns True if database exists and is not
    empty.
    """
    if clean:
        print("Cleaning database..")
        clean_database()
    return database_is_empty(dbsession)


def make_init_db(fpl_team_id, dbsession):
    make_team_table(dbsession=dbsession)
    make_fixture_table(dbsession=dbsession)
    make_result_table(dbsession=dbsession)
    make_fifa_ratings_table(dbsession=dbsession)

    make_player_table(dbsession=dbsession)
    make_attributes_table(dbsession=dbsession)
    make_playerscore_table(dbsession=dbsession)

    fill_initial_squad(fpl_team_id=fpl_team_id, dbsession=dbsession)

    print("DONE!")
    return not database_is_empty(dbsession)


def main():
    parser = argparse.ArgumentParser(description="Customise fpl team id")
    parser.add_argument(
        "--fpl_team_id", help="specify fpl team id", type=int, required=False
    )
    parser.add_argument(
        "--clean",
        help="If set, delete and re-create any pre-existing AIrsenal database",
        action="store_true",
    )
    args = parser.parse_args()

    with session_scope() as dbsession:
        continue_setup = check_clean_db(args.clean, dbsession)
        if continue_setup:
            make_init_db(args.fpl_team_id, dbsession)
        else:
            print(
                "AIrsenal database already exists. "
                "Run 'airsenal_setup_initial_db --clean' to delete and recreate it,\n"
                "or keep the current database and continue to 'airsenal_update_db'."
            )
