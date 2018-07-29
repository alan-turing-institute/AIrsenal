"""
test various methods of the Team class.
"""

import pytest

from ..team import Team
from ..player import Player


def test_cant_add_same_player():
    """
    can't add a player thats already on the team.
    """
    t = Team()
    added_ok = t.add_player(1)
    assert(added_ok)
    added_ok = t.add_player(1)
    assert(not added_ok)


def test_cant_add_too_many_per_position():
    """
    no more than two keepers, 5 defenders, 5 midfielders, 3 forwards.
    """
    t = Team()

def test_cant_add_too_many_per_team():
    """
    no more than three from the same team.
    """


def test_empty_team():
    """
    shouldn't be able to estimate points with
    no players.
    """
    t = Team()
    with pytest.raises(RuntimeError) as errmsg:
        t.get_expected_points(1)
    assert(str(errmsg.value) == "Team is incomplete")
