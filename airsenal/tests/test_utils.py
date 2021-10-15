"""
test some db access helper functions
"""

from airsenal.conftest import test_session_scope
from airsenal.framework.schema import Player
from airsenal.framework.utils import get_player, get_player_id, get_player_name


def test_get_player_name(fill_players):
    """
    Should be able to find a player with integer argument
    """
    with test_session_scope() as tsession:
        assert get_player_name(1, tsession) == "Bob"


def test_get_player_id(fill_players):
    """
    Should be able to find a player with string argument
    """
    with test_session_scope() as tsession:
        assert get_player_id("Bob", tsession) == 1


def test_get_player(fill_players):
    """
    test we can get a player object from either a name or an id
    """
    with test_session_scope() as tsession:
        p = get_player("Bob", tsession)
        assert isinstance(p, Player)
        assert p.player_id == 1
