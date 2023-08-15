"""
test that functions supporting our API work.
"""

import re

from airsenal.conftest import API_SESSION_ID, session_scope
from airsenal.framework.api_utils import (
    add_session_player,
    get_session_budget,
    get_session_players,
    list_players_teams_prices,
    remove_session_player,
    reset_session_squad,
    set_session_budget,
    validate_session_squad,
)
from airsenal.framework.schema import SessionBudget, SessionSquad


def test_reset_session_squad():
    with session_scope() as ts:
        assert reset_session_squad(session_id=API_SESSION_ID, dbsession=ts)
        st = ts.query(SessionSquad).filter_by(session_id=API_SESSION_ID).all()
        assert len(st) == 0
        sb = ts.query(SessionBudget).filter_by(session_id=API_SESSION_ID).all()
        assert len(sb) == 1
        assert sb[0].budget == 1000
        assert get_session_budget(API_SESSION_ID, ts) == 1000


def test_list_all_players(fill_players):
    with session_scope() as ts:
        player_list = list_players_teams_prices(dbsession=ts, gameweek=1)
        assert isinstance(player_list, list)
        assert len(player_list) > 0
        first_player = player_list[0]
        # test the format of the returned strings
        assert re.search(r"[a-zA-Z\s]+\([A-Z]{3}\)\: [\d\.]+", first_player)


def test_add_player(fill_players):
    with session_scope() as ts:
        assert reset_session_squad(session_id=API_SESSION_ID, dbsession=ts)
        assert add_session_player(12, API_SESSION_ID, ts)
        players = get_session_players(API_SESSION_ID, ts)
        assert len(players) == 1
        assert players[0]["id"] == 12


def test_cant_add_same_player_twice(fill_players):
    with session_scope() as ts:
        assert reset_session_squad(session_id=API_SESSION_ID, dbsession=ts)
        assert add_session_player(33, API_SESSION_ID, ts)
        assert not add_session_player(33, API_SESSION_ID, ts)
        players = get_session_players(API_SESSION_ID, ts)
        assert len(players) == 1
        assert players[0]["id"] == 33


def test_remove_player(fill_players):
    with session_scope() as ts:
        assert reset_session_squad(session_id=API_SESSION_ID, dbsession=ts)
        assert add_session_player(12, API_SESSION_ID, ts)
        assert remove_session_player(12, API_SESSION_ID, ts)
        players = get_session_players(API_SESSION_ID, ts)
        assert len(players) == 0


def test_get_budget():
    with session_scope() as ts:
        assert reset_session_squad(session_id=API_SESSION_ID, dbsession=ts)
        assert get_session_budget(API_SESSION_ID, ts) == 1000


def test_set__get_budget():
    with session_scope() as ts:
        assert reset_session_squad(session_id=API_SESSION_ID, dbsession=ts)
        assert set_session_budget(500, API_SESSION_ID, ts)
        assert get_session_budget(API_SESSION_ID, ts) == 500


def test_invalid_session_squad(fill_players):
    with session_scope() as ts:
        reset_session_squad(session_id=API_SESSION_ID, dbsession=ts)
        assert not validate_session_squad(API_SESSION_ID, ts)
        # add one player - check it is still invalid
        assert add_session_player(5, API_SESSION_ID, ts)
        assert not validate_session_squad(API_SESSION_ID, ts)


def test_valid_session_squad(fill_players):
    with session_scope() as ts:
        reset_session_squad(API_SESSION_ID, ts)
        for pid in range(15):
            assert add_session_player(pid, API_SESSION_ID, ts)
        assert validate_session_squad(API_SESSION_ID, ts)
