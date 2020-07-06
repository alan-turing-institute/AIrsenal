"""
test some db access helper functions
"""

import pytest

from .fixtures import test_session_scope, fill_players
from ..framework.utils import get_player_name, get_player_id


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


def get_player(fill_players):
    """
    test we can get a player object from either a name or an id
    """
    with test_session_scope() as tsession:
        p = get_player("Bob", tsession)
        assert isinstance(p, Player)
        assert p.player_id == 1
