"""
test some db access helper functions
"""

from airsenal.conftest import TEST_PAST_SEASON, past_data_session_scope, session_scope
from airsenal.framework.schema import Fixture, Player
from airsenal.framework.utils import (
    get_gameweek_by_date,
    get_last_complete_gameweek_in_db,
    get_player,
    get_player_id,
    get_player_name,
    get_return_gameweek_by_date,
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


def test_get_return_gameweek_by_date():
    with past_data_session_scope() as ts:
        gw = get_return_gameweek_by_date(
            "2020-09-18", "ARS", season=TEST_PAST_SEASON, dbsession=ts
        )
        assert gw == 2

        gw = get_return_gameweek_by_date(
            "2020-09-20T12:34:00Z", "ARS", season=TEST_PAST_SEASON, dbsession=ts
        )
        assert gw == 3


def test_get_gameweek_by_date():
    with past_data_session_scope() as ts:
        gw = get_gameweek_by_date(
            "2020-09-20T12:34:00Z", season=TEST_PAST_SEASON, dbsession=ts
        )
        assert gw == 2


def test_get_last_complete_gameweek_in_db():
    """Runs a query against a relationship rather than a column, which is easy to
    get wrong - it raised NotImplementedError at the start of a new season."""
    with past_data_session_scope() as ts:
        last_gw = get_last_complete_gameweek_in_db(TEST_PAST_SEASON, dbsession=ts)
    assert last_gw is None or isinstance(last_gw, int)


def test_get_last_complete_gameweek_with_no_results():
    """Fixtures scheduled but none played yet - where every new season starts.
    No gameweek is complete, so the answer is the one before the first."""
    with session_scope() as ts:
        for gw in (1, 2):
            fixture = Fixture()
            fixture.gameweek = gw
            fixture.season = "9999"
            fixture.tag = "test"
            fixture.home_team = "ARS"
            fixture.away_team = "CHE"
            ts.add(fixture)
        ts.flush()
        assert get_last_complete_gameweek_in_db("9999", dbsession=ts) == 0
