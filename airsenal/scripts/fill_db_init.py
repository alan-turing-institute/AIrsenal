"""Script to fill the database after install."""
import argparse
from typing import List

from sqlalchemy.orm.session import Session

from airsenal.framework.schema import clean_database, database_is_empty, session_scope
from airsenal.framework.season import CURRENT_SEASON, sort_seasons
from airsenal.framework.transaction_utils import fill_initial_squad
from airsenal.framework.utils import get_past_seasons
from airsenal.scripts.fill_absence_table import make_absence_table
from airsenal.scripts.fill_fifa_ratings_table import make_fifa_ratings_table
from airsenal.scripts.fill_fixture_table import make_fixture_table
from airsenal.scripts.fill_player_attributes_table import make_attributes_table
from airsenal.scripts.fill_player_table import make_player_table
from airsenal.scripts.fill_playerscore_table import make_playerscore_table
from airsenal.scripts.fill_result_table import make_result_table
from airsenal.scripts.fill_team_table import make_team_table


def check_clean_db(clean: bool, dbsession: Session) -> bool:
    """Check whether an AIrsenal database already exists. If clean is True attempt to
    delete any pre-existing database first. Returns True if database exists and is not
    empty.
    """
    if clean:
        print("Cleaning database..")
        clean_database()
    return database_is_empty(dbsession)


def make_init_db(fpl_team_id: int, seasons: List[str], dbsession: Session) -> bool:
    seasons = sort_seasons(seasons)
    make_team_table(seasons=seasons, dbsession=dbsession)
    make_fixture_table(seasons=seasons, dbsession=dbsession)
    make_result_table(seasons=seasons, dbsession=dbsession)
    make_fifa_ratings_table(seasons=seasons, dbsession=dbsession)

    make_player_table(seasons=seasons, dbsession=dbsession)
    make_attributes_table(seasons=seasons, dbsession=dbsession)
    make_playerscore_table(seasons=seasons, dbsession=dbsession)
    make_absence_table(seasons=seasons, dbsession=dbsession)

    if CURRENT_SEASON in seasons:
        fill_initial_squad(fpl_team_id=fpl_team_id, dbsession=dbsession)

    print("DONE!")
    return not database_is_empty(dbsession)


def check_positive_int(value: int) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue


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
    parser.add_argument(
        "--n_previous",
        help="specify how many seasons to look back into the past for (defaults to 3)",
        type=int,
        choices=range(1, int(CURRENT_SEASON[2:]) - 16 + 1),  # years since 1516 season
        default=3,
        required=False,
    )
    parser.add_argument(
        "--no_current_season",
        help="If set, does not include CURRENT_SEASON in database",
        action="store_true",
    )
    args = parser.parse_args()

    with session_scope() as dbsession:
        continue_setup = check_clean_db(args.clean, dbsession)
        if continue_setup:
            if args.no_current_season:
                seasons = get_past_seasons(args.n_previous)
            else:
                seasons = [CURRENT_SEASON] + get_past_seasons(args.n_previous)
            make_init_db(args.fpl_team_id, seasons, dbsession)
        else:
            print(
                "AIrsenal database already exists. "
                "Run 'airsenal_setup_initial_db --clean' to delete and recreate it,\n"
                "or keep the current database and continue to 'airsenal_update_db'."
            )
