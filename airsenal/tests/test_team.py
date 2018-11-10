"""
test various methods of the Team class.
"""

import pytest

from .fixtures import test_session_scope, fill_players
from ..framework.utils import get_player_name, get_player_id

from ..framework.team import Team
from ..framework.player import CandidatePlayer


def test_add_player_by_id(fill_players):
    """
    Should be able to add a player with integer argument
    """
    with test_session_scope() as ts:
        t = Team()
        added_ok = t.add_player(50,dbsession=ts)
        assert added_ok


def test_add_player_by_name():
    """
    Should be able to add a player with string argument
    """
    with test_session_scope() as ts:
        t = Team()
        added_ok = t.add_player("Alice", dbsession=ts)
        assert added_ok


def test_cant_add_same_player():
    """
    can't add a player thats already on the team.
    """
    t = Team()
    added_ok = t.add_player(1)
    assert added_ok
    added_ok = t.add_player(1)
    assert not added_ok


def test_cant_add_too_many_per_position():
    """
    no more than two keepers, 5 defenders, 5 midfielders, 3 forwards.
    """
    t = Team()
    # keepers
    assert t.add_player("Alice")
    assert t.add_player("Bob")
    assert not t.add_player("Pedro")
    # defenders
    assert t.add_player("Carla")
    assert t.add_player("Donald")
    assert t.add_player("Erica")
    assert t.add_player("Frank")
    assert t.add_player("Gerry")
    assert not t.add_player("Stefan")


def test_cant_add_too_many_per_team():
    """
    no more than three from the same team.
    """
    t = Team()
    assert t.add_player(1)
    assert t.add_player(2)
    assert t.add_player(3)
    assert not t.add_player(4)


def test_cant_exceed_budget():
    """
    try and make an expensive team
    """
    t = Team()
    added_ok = True
    added_ok = added_ok and t.add_player("Harry Kane")
    added_ok = added_ok and t.add_player("Romelu Lukaku")
    added_ok = added_ok and t.add_player("Roberto Firmino")
    added_ok = added_ok and t.add_player("Mohamed Salah")
    added_ok = added_ok and t.add_player("Raheem Sterling")
    added_ok = added_ok and t.add_player("Eden Hazard")
    added_ok = added_ok and t.add_player("Kevin De Bruyne")
    added_ok = added_ok and t.add_player("Riyad Mahrez")
    added_ok = added_ok and t.add_player("Marcos Alonso")
    added_ok = added_ok and t.add_player("Chris Smalling")
    added_ok = added_ok and t.add_player("Victor Moses")
    added_ok = added_ok and t.add_player("Antonio Valencia")
    added_ok = added_ok and t.add_player("Serge Aurier")
    added_ok = added_ok and t.add_player("Hugo Lloris")
    added_ok = added_ok and t.add_player("Petr Cech")

    assert not added_ok


def test_remove_player():
    """
    add a player then remove them.
    """
    t = Team()
    t.add_player(1)
    assert len(t.players) == 1
    assert t.num_position["GK"] == 1
    t.remove_player(1)
    assert len(t.players) == 0
    assert t.num_position["GK"] == 0
    assert t.budget == 1000


def test_empty_team():
    """
    shouldn't be able to estimate points with
    no players.
    """
    t = Team()
    with pytest.raises(RuntimeError) as errmsg:
        t.get_expected_points(1,"dummy")
    assert str(errmsg.value) == "Team is incomplete"
