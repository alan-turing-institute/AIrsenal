"""
test some db access helper functions
"""

from airsenal.conftest import TEST_PAST_SEASON, past_data_session_scope, session_scope
from airsenal.framework.schema import Player
from airsenal.framework.utils import (
    get_gameweek_by_fixture_date,
    get_next_gameweek_by_date,
    get_player,
    get_player_id,
    get_player_name,
)


def test_get_player_name(fill_players):
    """
    Should be able to find a player with integer argument
    """
    with session_scope() as tsession:
        assert get_player_name(1, tsession) == "Bob"


def test_get_player_id(fill_players):
    """
    Should be able to find a player with string argument
    """
    with session_scope() as tsession:
        assert get_player_id("Bob", tsession) == 1


def test_get_player(fill_players):
    """
    test we can get a player object from either a name or an id
    """
    with session_scope() as tsession:
        p = get_player("Bob", tsession)
        assert isinstance(p, Player)
        assert p.player_id == 1


def test_get_next_gameweek_by_date():
    with past_data_session_scope() as ts:
        gw = get_next_gameweek_by_date(
            "2020-09-18", season=TEST_PAST_SEASON, dbsession=ts
        )
        assert gw == 2

        gw = get_next_gameweek_by_date(
            "2020-09-20T12:34:00Z", season=TEST_PAST_SEASON, dbsession=ts
        )
        assert gw == 3


def test_get_gameweek_by_fixture_date():
    with past_data_session_scope() as ts:
        gw = get_gameweek_by_fixture_date(
            "2020-09-18", season=TEST_PAST_SEASON, dbsession=ts
        )
        assert gw is None

        gw = get_gameweek_by_fixture_date(
            "2020-09-20T12:34:00Z", season=TEST_PAST_SEASON, dbsession=ts
        )
        assert gw == 2
