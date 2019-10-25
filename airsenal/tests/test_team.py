"""
test various methods of the Team class.
"""

import pytest

from .fixtures import test_session_scope, fill_players
from ..framework.utils import get_player_name, get_player_id

from ..framework.team import Team
from ..framework.player import CandidatePlayer

TEST_SEASON="1920"

def test_add_player_by_id(fill_players):
    """
    Should be able to add a player with integer argument
    """
    with test_session_scope() as ts:
        t = Team()
        added_ok = t.add_player(50,season=TEST_SEASON,dbsession=ts)
        assert added_ok


def test_add_player_by_name(fill_players):
    """
    Should be able to add a player with string argument
    """
    with test_session_scope() as ts:
        t = Team()
        added_ok = t.add_player("Alice",season=TEST_SEASON,
                                dbsession=ts)
        assert added_ok


def test_cant_add_same_player(fill_players):
    """
    can't add a player thats already on the team.
    """
    with test_session_scope() as ts:
        t = Team()
        added_ok = t.add_player(1,season=TEST_SEASON,dbsession=ts)
        assert added_ok
        added_ok = t.add_player(1,season=TEST_SEASON,dbsession=ts)
        assert not added_ok


def test_cant_add_too_many_per_position(fill_players):
    """
    no more than two keepers, 5 defenders, 5 midfielders, 3 forwards.
    """
    with test_session_scope() as ts:
        t = Team()
        # keepers
        assert t.add_player("Alice",season=TEST_SEASON,dbsession=ts)
        assert t.add_player("Bob",season=TEST_SEASON,dbsession=ts)
        assert not t.add_player("Pedro",season=TEST_SEASON,dbsession=ts)
        # defenders
        assert t.add_player("Carla",season=TEST_SEASON,dbsession=ts)
        assert t.add_player("Donald",season=TEST_SEASON,dbsession=ts)
        assert t.add_player("Erica",season=TEST_SEASON,dbsession=ts)
        assert t.add_player("Frank",season=TEST_SEASON,dbsession=ts)
        assert t.add_player("Gerry",season=TEST_SEASON,dbsession=ts)
        assert not t.add_player("Stefan",season=TEST_SEASON,dbsession=ts)


def test_cant_add_too_many_per_team(fill_players):
    """
    no more than three from the same team.
    """
    with test_session_scope() as ts:
        t = Team()
        assert t.add_player(1,season=TEST_SEASON,dbsession=ts)
        assert t.add_player(21,season=TEST_SEASON,dbsession=ts)
        assert t.add_player(41,season=TEST_SEASON,dbsession=ts)
        assert not t.add_player(61,season=TEST_SEASON,dbsession=ts)


def test_cant_exceed_budget():
    """
    try and make an expensive team
    """
    with test_session_scope() as ts:
        t = Team()
        added_ok = True
        added_ok = added_ok and t.add_player(45,season=TEST_SEASON,dbsession=ts)
        added_ok = added_ok and t.add_player(46,season=TEST_SEASON,dbsession=ts)
        added_ok = added_ok and t.add_player(47,season=TEST_SEASON,dbsession=ts)
        added_ok = added_ok and t.add_player(48,season=TEST_SEASON,dbsession=ts)
        added_ok = added_ok and t.add_player(49,season=TEST_SEASON,dbsession=ts)
        added_ok = added_ok and t.add_player(50,season=TEST_SEASON,dbsession=ts)
        added_ok = added_ok and t.add_player(51,season=TEST_SEASON,dbsession=ts)
        added_ok = added_ok and t.add_player(52,season=TEST_SEASON,dbsession=ts)
        added_ok = added_ok and t.add_player(53,season=TEST_SEASON,dbsession=ts)
        added_ok = added_ok and t.add_player(54,season=TEST_SEASON,dbsession=ts)
        added_ok = added_ok and t.add_player(55,season=TEST_SEASON,dbsession=ts)
        added_ok = added_ok and t.add_player(56,season=TEST_SEASON,dbsession=ts)
        added_ok = added_ok and t.add_player(57,season=TEST_SEASON,dbsession=ts)
        added_ok = added_ok and t.add_player(58,season=TEST_SEASON,dbsession=ts)
        added_ok = added_ok and t.add_player(59,season=TEST_SEASON,dbsession=ts)
        assert not added_ok


def test_remove_player(fill_players):
    """
    add a player then remove them.
    """
    with test_session_scope() as ts:
        t = Team()
        t.add_player(1,season=TEST_SEASON,dbsession=ts)
        assert len(t.players) == 1
        assert t.num_position["GK"] == 1
        t.remove_player(1, use_api=False)
        assert len(t.players) == 0
        assert t.num_position["GK"] == 0
        assert t.budget == 1000


def test_empty_team(fill_players):
    """
    shouldn't be able to estimate points with
    no players.
    """
    t = Team()
    with pytest.raises(RuntimeError) as errmsg:
        t.get_expected_points(1,"dummy")
    assert str(errmsg.value) == "Team is incomplete"
