"""
test various methods of the Team class.
"""

import pytest

from ..team import Team
from ..player import Player


def test_add_player_by_id():
    """
    Should be able to add a player with integer argument
    """
    t = Team()
    added_ok = t.add_player(50)
    assert added_ok


def test_add_player_by_name():
    """
    Should be able to add a player with string argument
    """
    t = Team()
    added_ok = t.add_player("Raheem Sterling")
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
    assert t.add_player("Jordan Pickford")
    assert t.add_player("Claudio Bravo")
    assert not t.add_player("Mathew Ryan")
    # defenders
    assert t.add_player("Scott Malone")
    assert t.add_player("Winston Reid")
    assert t.add_player("Younes Kaboul")
    assert t.add_player("Scott Dann")
    assert t.add_player("Mason Holgate")
    assert not t.add_player("Lewis Dunk")


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
        t.get_expected_points(1)
    assert str(errmsg.value) == "Team is incomplete"
