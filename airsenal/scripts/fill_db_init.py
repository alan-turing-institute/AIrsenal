"""Script to fill the database after install."""
from .fill_team_table import make_team_table
from .fill_player_table import make_player_table
from .fill_fixture_table import make_fixture_table
from .fill_result_table import make_result_table
from .fill_playerscore_table import make_playerscore_table
from .fill_fifa_ratings_table import make_fifa_ratings_table
from .fill_predictedscore_table import make_predictedscore_table

from ..framework.transaction_utils import fill_initial_team
from ..framework.schema import session_scope


def main():

    with session_scope() as session:
        make_team_table(session)
        make_player_table(session)
        make_fixture_table(session)
        make_result_table(session)
        make_playerscore_table(session)
        make_fifa_ratings_table(session)
        fill_initial_team(session)
