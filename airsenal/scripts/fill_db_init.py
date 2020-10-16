"""Script to fill the database after install."""
from airsenal.scripts.fill_team_table import make_team_table
from airsenal.scripts.fill_player_table import make_player_table
from airsenal.scripts.fill_player_attributes_table import make_attributes_table
from airsenal.scripts.fill_fixture_table import make_fixture_table
from airsenal.scripts.fill_result_table import make_result_table
from airsenal.scripts.fill_playerscore_table import make_playerscore_table
from airsenal.scripts.fill_fifa_ratings_table import make_fifa_ratings_table

from airsenal.framework.transaction_utils import fill_initial_team
from airsenal.framework.schema import session_scope


def main():

    with session_scope() as session:
        make_team_table(session)
        make_fixture_table(session)
        make_result_table(session)
        make_fifa_ratings_table(session)

        make_player_table(session)
        make_attributes_table(session)
        make_playerscore_table(session)

        fill_initial_team(session)

        print("DONE!")