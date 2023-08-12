"""
test various methods of the Team class.
"""

import pytest

from airsenal.conftest import session_scope
from airsenal.framework.squad import Squad
from airsenal.framework.utils import CURRENT_SEASON

TEST_SEASON = CURRENT_SEASON


def test_add_player_by_id(fill_players):
    """
    Should be able to add a player with integer argument
    """
    with session_scope() as ts:
        t = Squad(season=TEST_SEASON)
        added_ok = t.add_player(50, dbsession=ts)
        assert added_ok


def test_add_player_by_name(fill_players):
    """
    Should be able to add a player with string argument
    """
    with session_scope() as ts:
        t = Squad(season=TEST_SEASON)
        added_ok = t.add_player("Alice", dbsession=ts)
        assert added_ok


def test_cant_add_same_player(fill_players):
    """
    can't add a player thats already on the squad.
    """
    with session_scope() as ts:
        t = Squad(season=TEST_SEASON)
        added_ok = t.add_player(1, dbsession=ts)
        assert added_ok
        added_ok = t.add_player(1, dbsession=ts)
        assert not added_ok


def test_cant_add_too_many_per_position(fill_players):
    """
    no more than two keepers, 5 defenders, 5 midfielders, 3 forwards.
    """
    with session_scope() as ts:
        t = Squad(season=TEST_SEASON)
        # keepers
        assert t.add_player("Alice", dbsession=ts)
        assert t.add_player("Bob", dbsession=ts)
        assert not t.add_player("Pedro", dbsession=ts)
        # defenders
        assert t.add_player("Carla", dbsession=ts)
        assert t.add_player("Donald", dbsession=ts)
        assert t.add_player("Erica", dbsession=ts)
        assert t.add_player("Frank", dbsession=ts)
        assert t.add_player("Gerry", dbsession=ts)
        assert not t.add_player("Stefan", dbsession=ts)


def test_cant_add_too_many_per_squad(fill_players):
    """
    no more than three from the same squad.
    """
    with session_scope() as ts:
        t = Squad(season=TEST_SEASON)
        assert t.add_player(1, dbsession=ts)
        assert t.add_player(21, dbsession=ts)
        assert t.add_player(41, dbsession=ts)
        assert not t.add_player(61, dbsession=ts)


def test_cant_exceed_budget():
    """
    try and make an expensive squad
    """
    with session_scope() as ts:
        t = Squad(season=TEST_SEASON)
        added_ok = True
        added_ok = added_ok and t.add_player(45, dbsession=ts)
        added_ok = added_ok and t.add_player(46, dbsession=ts)
        added_ok = added_ok and t.add_player(47, dbsession=ts)
        added_ok = added_ok and t.add_player(48, dbsession=ts)
        added_ok = added_ok and t.add_player(49, dbsession=ts)
        added_ok = added_ok and t.add_player(50, dbsession=ts)
        added_ok = added_ok and t.add_player(51, dbsession=ts)
        added_ok = added_ok and t.add_player(52, dbsession=ts)
        added_ok = added_ok and t.add_player(53, dbsession=ts)
        added_ok = added_ok and t.add_player(54, dbsession=ts)
        added_ok = added_ok and t.add_player(55, dbsession=ts)
        added_ok = added_ok and t.add_player(56, dbsession=ts)
        added_ok = added_ok and t.add_player(57, dbsession=ts)
        added_ok = added_ok and t.add_player(58, dbsession=ts)
        added_ok = added_ok and t.add_player(59, dbsession=ts)
        assert not added_ok


def test_remove_player(fill_players):
    """
    add a player then remove them.
    """
    with session_scope() as ts:
        t = Squad(season=TEST_SEASON)
        t.add_player(1, dbsession=ts)
        assert len(t.players) == 1
        assert t.num_position["GK"] == 1
        t.remove_player(1, use_api=False, dbsession=ts)
        assert len(t.players) == 0
        assert t.num_position["GK"] == 0
        assert t.budget == 1000


def test_empty_squad(fill_players):
    """
    shouldn't be able to estimate points with
    no players.
    """
    t = Squad()
    with pytest.raises(RuntimeError) as errmsg:
        t.get_expected_points(1, "dummy")
    assert str(errmsg.value) == "Squad is incomplete"


def test_order_substitutes():
    t = Squad()

    class MockPlayer:
        def __init__(self, points, is_starting, name, squad):
            self.predicted_points = {0: {0: points}}
            self.is_starting = is_starting
            self.name = name
            self.squad = squad
            self.sub_position = None

    players = [
        MockPlayer(10, False, "a", "A"),
        MockPlayer(9, False, "b", "B"),
        MockPlayer(8, False, "c", "C"),
        MockPlayer(11, True, "d", "D"),
    ]

    t.players = players
    t.order_substitutes(0, 0)

    expected_sub_positions = [0, 1, 2, None]
    for player, sub_position in zip(players, expected_sub_positions):
        assert player.sub_position == sub_position

    # test the logic that's use in __repr__ as well
    subs = [p for p in t.players if not p.is_starting]
    subs.sort(key=lambda p: p.sub_position)
    expected_names = ["a", "b", "c"]
    for player, expected_name in zip(subs, expected_names):
        assert player.name == expected_name


def test_get_expected_points():
    t = Squad()

    class MockPlayer:
        def __init__(
            self,
            name,
            squad,
            position,
            points,
            is_starting,
            is_captain,
            is_vice_captain,
        ):
            self.name = name
            self.squad = squad
            self.position = position
            self.predicted_points = {0: {0: points}}
            self.is_starting = is_starting
            self.sub_position = None
            self.is_captain = is_captain
            self.is_vice_captain = is_vice_captain

        def calc_predicted_points(self, tag):
            pass

    # 3 pts captain (x2 = 6pts, or x3 = 9pts for TC)
    # 2 pts starters
    # 1 pt subs
    players = [
        MockPlayer("a", "A", "GK", 2, True, False, False),
        MockPlayer("b", "B", "GK", 1, False, False, False),  # sub 1
        MockPlayer("c", "C", "DEF", 2, True, False, False),
        MockPlayer("d", "D", "DEF", 2, True, False, False),
        MockPlayer("e", "E", "DEF", 2, True, False, False),
        MockPlayer("f", "F", "DEF", 1, False, False, False),  # sub 2
        MockPlayer("g", "G", "DEF", 1, False, False, False),  # sub 3
        MockPlayer("h", "H", "MID", 2, True, False, False),
        MockPlayer("i", "I", "MID", 2, True, False, False),
        MockPlayer("j", "J", "MID", 2, True, False, False),
        MockPlayer("k", "K", "MID", 2, True, False, False),
        MockPlayer("l", "L", "MID", 1, False, False, False),  # sub 4
        MockPlayer("m", "M", "FWD", 3, True, True, False),  # captain
        MockPlayer("n", "N", "FWD", 2, True, False, True),  # vice-captain
        MockPlayer("o", "O", "FWD", 2, True, False, False),
    ]

    t.players = players
    t.num_position = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}

    # no chips
    assert t.get_expected_points(0, 0) == 26
    # bench boost
    assert t.get_expected_points(0, 0, bench_boost=True) == 30
    # triple captain
    assert t.get_expected_points(0, 0, triple_captain=True) == 29
