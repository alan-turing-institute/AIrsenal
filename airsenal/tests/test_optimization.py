"""
Test the optimization of transfers, generating a few simplified scenarios
and checking that the optimizer finds the expected outcome.
"""
from operator import itemgetter
from unittest import mock

import pytest

from airsenal.framework.optimization_transfers import (
    make_optimum_double_transfer,
    make_optimum_single_transfer,
)
from airsenal.framework.optimization_utils import (
    count_expected_outputs,
    get_discount_factor,
    next_week_transfers,
)
from airsenal.framework.squad import Squad

pytestmark = pytest.mark.filterwarnings("ignore:Using purchase price as sale price")


class DummyPlayer(object):
    """
    fake player that we can add to a squad, giving a specified expected score.
    """

    def __init__(self, player_id, position, points_dict):
        """
        we generate squad to avoid >3-players-per-team problem,
        and set price to 0 to avoid overrunning budget.
        """
        self.player_id = player_id
        self.fpl_api_id = player_id
        self.name = f"player_{player_id}"
        self.position = position
        self.team = f"DUMMY_TEAM_{player_id}"
        self.purchase_price = 0
        self.is_starting = True
        self.is_captain = False
        self.is_vice_captain = False
        self.predicted_points = {"DUMMY": points_dict}
        self.sub_position = None

    def calc_predicted_points(self, dummy):
        pass


def generate_dummy_squad(player_points_dict=None):
    """
    Fill a squad up with dummy players.
    player_points_dict is a dictionary
    { player_id: { gw: points,...} ,...}
    """
    if not player_points_dict:  # make a simple one
        player_points_dict = {i: {1: 2} for i in range(15)}
    t = Squad()
    for i in range(15):
        if i < 2:
            position = "GK"
        elif i < 7:
            position = "DEF"
        elif i < 12:
            position = "MID"
        else:
            position = "FWD"
        t.add_player(DummyPlayer(i, position, player_points_dict[i]))
    return t


def predicted_point_mock_generator(point_dict):
    """
    return a function that will mock the get_predicted_points function
    the point_dict it is given should be keyed by position, i.e.
    {"GK" : {player_id: points, ...}, "DEF": {}, ... }
    """

    def mock_get_predicted_points(
        gameweek, tag, position, team=None, season=None, dbsession=None
    ):
        """
        return an ordered list in the same way as the real
        get_predicted_points func does. EXCEPT - we return dummy players rather
        than just ids (so the Squad.add_player can add them)
        """
        output_pid_list = [(k, v) for k, v in point_dict[position].items()]
        output_pid_list.sort(key=itemgetter(1), reverse=True)
        #        return output_pid_list
        if isinstance(gameweek, list):
            gameweek = gameweek[0]
        return [
            (DummyPlayer(entry[0], position, {gameweek: entry[1]}), entry[1])
            for entry in output_pid_list
        ]

    return mock_get_predicted_points


def test_subs():
    """
    mock squads with some players predicted some points, and
    some predicted to score zero, and check we get the right starting 11.
    """
    points_dict = {
        0: {1: 0},
        1: {1: 2},
        2: {1: 2},
        3: {1: 2},
        4: {1: 0},
        5: {1: 2},
        6: {1: 2},
        7: {1: 2},
        8: {1: 2},
        9: {1: 0},
        10: {1: 2},
        11: {1: 4},
        12: {1: 0},
        13: {1: 2},
        14: {1: 3},
    }
    # should get 4,4,2, with players 0,4,9,12 on the bench,
    # captain player 11, vice-captain player 14
    # should have 29 points (9*2 + 3 + (2*4) )
    t = generate_dummy_squad(points_dict)
    ep = t.get_expected_points(1, "DUMMY")
    assert ep == 29
    assert t.players[0].is_starting is False
    assert t.players[4].is_starting is False
    assert t.players[9].is_starting is False
    assert t.players[12].is_starting is False
    assert t.players[11].is_captain is True
    assert t.players[14].is_vice_captain is True


def test_single_transfer():
    """
    mock squad with all players predicted 2 points, and potential transfers
    with higher scores, check we get the best transfer.
    """
    t = generate_dummy_squad()
    position_points_dict = {
        "GK": {0: 2, 1: 2, 100: 0, 101: 0, 200: 3, 201: 2},  # in the orig squad
        "DEF": {
            2: 2,
            3: 2,
            4: 2,
            5: 2,
            6: 2,  # in the orig squad
            103: 0,
            104: 0,
            105: 5,
            106: 2,
            107: 2,
            203: 0,
            204: 0,
            205: 1,
            206: 2,
            207: 2,
        },
        "MID": {
            7: 2,
            8: 2,
            9: 2,
            10: 2,
            11: 2,  # in the orig squad
            108: 2,
            109: 2,
            110: 3,
            111: 3,
            112: 0,
            208: 2,
            209: 2,
            210: 3,
            211: 3,
            212: 0,
        },
        "FWD": {12: 2, 13: 2, 14: 2, 113: 6, 114: 3, 115: 7},  # in the orig squad
    }
    mock_pred_points = predicted_point_mock_generator(position_points_dict)

    with mock.patch(
        "airsenal.framework.optimization_transfers.get_predicted_points",
        side_effect=mock_pred_points,
    ):
        new_squad, pid_out, pid_in = make_optimum_single_transfer(t, "DUMMY", [1])
        # we should expect - player 115 to be transfered in, and to be captain.
    assert pid_in[0] == 115
    for p in new_squad.players:
        if p.player_id == 115:
            assert p.is_captain is True
        else:
            assert p.is_captain is False
    # expected points should be 10*2 + 7*2 = 34
    assert new_squad.get_expected_points(1, "DUMMY") == 34


def test_double_transfer():
    """
    mock squad with two players predicted low score, see if we get better players
    transferred in.
    """
    t = generate_dummy_squad()
    position_points_dict = {
        "GK": {0: 2, 1: 2, 100: 0, 101: 0, 200: 3, 201: 7},  # in the orig squad
        "DEF": {
            2: 2,
            3: 2,
            2: 2,
            5: 2,
            6: 2,  # in the orig squad
            103: 0,
            104: 0,
            105: 5,
            106: 2,
            107: 2,
            203: 0,
            204: 0,
            205: 1,
            206: 2,
            207: 2,
        },
        "MID": {
            7: 2,
            8: 2,
            9: 2,
            10: 2,
            11: 2,  # in the orig squad
            108: 2,
            109: 2,
            110: 3,
            111: 3,
            112: 0,
            208: 2,
            209: 2,
            210: 3,
            211: 3,
            212: 0,
        },
        "FWD": {12: 2, 13: 2, 14: 2, 113: 6, 114: 3, 115: 8},  # in the orig squad
    }
    mock_pred_points = predicted_point_mock_generator(position_points_dict)

    with mock.patch(
        "airsenal.framework.optimization_transfers.get_predicted_points",
        side_effect=mock_pred_points,
    ):
        new_squad, pid_out, pid_in = make_optimum_double_transfer(t, "DUMMY", [1])
        # we should expect 201 and 115 to be transferred in, and 1,15 to
        # be transferred out.   115 should be captain
        assert 201 in pid_in
        assert 115 in pid_in
        print(new_squad)
        for p in new_squad.players:
            if p.player_id == 115:
                assert p.is_captain is True
            else:
                assert p.is_captain is False


def test_get_discount_factor():
    """
    Discount factor discounts future gameweek score predictions based on the
    number of gameweeks ahead. It uses two discount types based on a discount
    of 14/15, exponential ({14/15}^{weeks ahead}) and constant
    (1-{14/15}*weeks ahead)
    """

    assert get_discount_factor(1, 4) == (14 / 15) ** (4 - 1)
    assert get_discount_factor(1, 4, "constant") == 1 - ((1 / 15) * (4 - 1))
    assert get_discount_factor(1, 20, "const") == 0
    assert get_discount_factor(1, 1, "const") == 1
    assert get_discount_factor(1, 1, "exp") == 1


def test_next_week_transfers_no_chips_no_constraints():
    # First week (blank starting strat with 1 free transfer available)
    strat = (1, 0, {"players_in": {}, "chips_played": {}})
    # No chips or constraints
    actual = next_week_transfers(
        strat,
        max_total_hit=None,
        allow_unused_transfers=True,
        max_transfers=2,
    )
    # (no. transfers, free transfers following week, points hit)
    expected = [(0, 2, 0), (1, 1, 0), (2, 1, 4)]
    assert actual == expected


def test_next_week_transfers_any_chip_no_constraints():
    # All chips, no constraints
    strat = (1, 0, {"players_in": {}, "chips_played": {}})
    actual = next_week_transfers(
        strat,
        max_total_hit=None,
        max_transfers=2,
        chips={
            "chips_allowed": ["wildcard", "free_hit", "bench_boost", "triple_captain"],
            "chip_to_play": None,
        },
    )
    expected = [
        (0, 2, 0),
        (1, 1, 0),
        (2, 1, 4),
        ("W", 1, 0),
        ("F", 1, 0),
        ("B0", 2, 0),
        ("B1", 1, 0),
        ("B2", 1, 4),
        ("T0", 2, 0),
        ("T1", 1, 0),
        ("T2", 1, 4),
    ]
    assert actual == expected


def test_next_week_transfers_no_chips_zero_hit():
    # No points hits
    strat = (1, 0, {"players_in": {}, "chips_played": {}})
    actual = next_week_transfers(
        strat,
        max_total_hit=0,
        allow_unused_transfers=True,
        max_transfers=2,
    )
    expected = [(0, 2, 0), (1, 1, 0)]
    assert actual == expected


def test_next_week_transfers_2ft_no_unused():
    # 2 free transfers available, no wasted transfers
    strat = (2, 0, {"players_in": {}, "chips_played": {}})
    actual = next_week_transfers(
        strat,
        max_total_hit=None,
        allow_unused_transfers=False,
        max_transfers=2,
    )
    expected = [(1, 2, 0), (2, 1, 0)]
    assert actual == expected


def test_next_week_transfers_chips_already_used():
    # Chips allowed but previously used
    strat = (
        1,
        0,
        {
            "players_in": {},
            "chips_played": {
                1: "wildcard",
                2: "free_hit",
                3: "bench_boost",
                4: "triple_captain",
            },
        },
    )
    actual = next_week_transfers(
        strat,
        max_total_hit=None,
        max_transfers=2,
    )
    expected = [(0, 2, 0), (1, 1, 0), (2, 1, 4)]
    assert actual == expected


def test_next_week_transfers_play_wildcard():
    strat = (1, 0, {"players_in": {}, "chips_played": {}})
    actual = next_week_transfers(
        strat,
        max_total_hit=None,
        max_transfers=2,
        chips={"chips_allowed": [], "chip_to_play": "wildcard"},
    )
    expected = [("W", 1, 0)]
    assert actual == expected


def test_next_week_transfers_2ft_allow_wildcard():
    strat = (2, 0, {"players_in": {}, "chips_played": {}})
    actual = next_week_transfers(
        strat,
        max_total_hit=None,
        max_transfers=2,
        chips={"chips_allowed": ["wildcard"], "chip_to_play": None},
    )
    expected = [(0, 2, 0), (1, 2, 0), (2, 1, 0), ("W", 1, 0)]
    assert actual == expected


def test_next_week_transfers_2ft_allow_wildcard_no_unused():
    strat = (2, 0, {"players_in": {}, "chips_played": {}})
    actual = next_week_transfers(
        strat,
        max_total_hit=None,
        allow_unused_transfers=False,
        max_transfers=2,
        chips={"chips_allowed": ["wildcard"], "chip_to_play": None},
    )
    expected = [(1, 2, 0), (2, 1, 0), ("W", 1, 0)]
    assert actual == expected


def test_next_week_transfers_2ft_play_wildcard():
    strat = (2, 0, {"players_in": {}, "chips_played": {}})
    actual = next_week_transfers(
        strat,
        max_total_hit=None,
        max_transfers=2,
        chips={"chips_allowed": [], "chip_to_play": "wildcard"},
    )
    expected = [("W", 1, 0)]
    assert actual == expected


def test_next_week_transfers_2ft_play_bench_boost_no_unused():
    strat = (2, 0, {"players_in": {}, "chips_played": {}})
    actual = next_week_transfers(
        strat,
        max_total_hit=None,
        allow_unused_transfers=False,
        max_transfers=2,
        chips={"chips_allowed": [], "chip_to_play": "bench_boost"},
    )
    expected = [("B1", 2, 0), ("B2", 1, 0)]
    assert actual == expected


def test_next_week_transfers_play_triple_captain_max_transfers_3():
    strat = (1, 0, {"players_in": {}, "chips_played": {}})
    actual = next_week_transfers(
        strat,
        max_total_hit=None,
        allow_unused_transfers=True,
        max_transfers=3,
        chips={"chips_allowed": [], "chip_to_play": "triple_captain"},
    )
    expected = [("T0", 2, 0), ("T1", 1, 0), ("T2", 1, 4), ("T3", 1, 8)]
    assert actual == expected


def test_count_expected_outputs_no_chips_no_constraints():
    # No constraints or chips, expect 3**num_gameweeks strategies
    count = count_expected_outputs(
        3,
        free_transfers=1,
        max_total_hit=None,
        allow_unused_transfers=True,
        next_gw=1,
        max_transfers=2,
        chip_gw_dict={},
    )
    assert count == 3**3

    # Max hit 0
    # Include:
    # (0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0), (0, 1, 1), (0, 1, 2),
    # (0, 2, 0), (0, 2, 1), (1, 0, 0), (1, 0, 1), (1, 0, 2), (1, 1, 0), (1, 1, 1)
    # Exclude:
    # (0, 2, 2), (1, 1, 2), (1, 2, 0), (1, 2, 1), (1, 2, 2), (2, 0, 0), (2, 0, 1),
    # (2, 0, 2), (2, 1, 0), (2, 1, 1), (2, 1, 2), (2, 2, 0), (2, 2, 1), (2, 2, 2)


def test_count_expected_outputs_no_chips_zero_hit():
    count = count_expected_outputs(
        3,
        free_transfers=1,
        max_total_hit=0,
        next_gw=1,
        max_transfers=2,
        chip_gw_dict={},
    )
    assert count == 13

    # Start with 2 FT and no unused
    # Include:
    # (0, 0, 0), (1, 1, 1), (1, 1, 2), (1, 2, 0), (1, 2, 1), (1, 2, 2), (2, 0, 1),
    # (2, 0, 2), (2, 1, 0), (2, 1, 1), (2, 1, 2), (2, 2, 0), (2, 2, 1), (2, 2, 2)
    # Exclude:
    # (0, 0, 1), (0, 0, 2), (0, 1, 0), (0, 1, 1), (0, 1, 2), (0, 2, 0), (0, 2, 1),
    # (0, 2, 2), (1, 0, 0), (1, 0, 1), (1, 0, 2), (1, 1, 0), (2, 0, 0)


def test_count_expected_outputs_no_chips_2ft_no_unused():
    count = count_expected_outputs(
        3,
        free_transfers=2,
        max_total_hit=None,
        allow_unused_transfers=False,
        next_gw=1,
        max_transfers=2,
    )
    assert count == 14

    # Wildcard, 2 weeks, no constraints
    # Strategies:
    # (0, 0), (0, 1), (0, 2), (0, 'W'), (1, 0), (1, 1), (1, 2), (1, 'W'), (2, 0),
    # (2, 1), (2, 2), (2, 'W'), ('W', 0), ('W', 1), ('W', 2)


def test_count_expected_wildcard_allowed_no_constraints():
    count = count_expected_outputs(
        2,
        free_transfers=1,
        max_total_hit=None,
        allow_unused_transfers=True,
        next_gw=1,
        max_transfers=2,
        chip_gw_dict={
            1: {"chips_allowed": ["wildcard"]},
            2: {"chips_allowed": ["wildcard"]},
            3: {"chips_allowed": ["wildcard"]},
        },
    )
    assert count == 15

    # Bench boost, 2 weeks, no constraints
    # Strategies:
    # (0, 0), (0, 1), (0, 2), (0, 'B0'), (0, 'B1'), (0, 'B2'), (1, 0), (1, 1), (1, 2),
    # (1, 'B0'), (1, 'B1'), (1, 'B2'), (2, 0), (2, 1), (2, 2), (2, 'B0'), (2, 'B1'),
    # (2, 'B2'), ('B0', 0), ('B0', 1), ('B0', 2), ('B1', 0), ('B1', 1), ('B1', 2),
    # ('B2', 0), ('B2', 1), ('B2', 2),


def count_expected_bench_boost_allowed_no_constraints():
    count = count_expected_outputs(
        2,
        free_transfers=1,
        max_total_hit=None,
        allow_unused_transfers=True,
        next_gw=1,
        max_transfers=2,
        chip_gw_dict={
            1: {"chips_allowed": ["bench_boost"]},
            2: {"chips_allowed": ["bench_boost"]},
            3: {"chips_allowed": ["bench_boost"]},
        },
    )
    assert count == 27

    # Force playing wildcard in first week
    # Strategies:
    # ("W",0), ("W,1), ("W",2)


def count_expected_play_wildcard_no_constraints():
    count = count_expected_outputs(
        2,
        free_transfers=1,
        max_total_hit=None,
        allow_unused_transfers=True,
        next_gw=1,
        max_transfers=2,
        chip_gw_dict={
            1: {"chip_to_play": "wildcard", "chips_allowed": []},
            2: {"chip_to_play": None, "chips_allowed": []},
        },
    )
    assert count == 3

    # Force playing free hit in first week, 2FT, don't allow unused
    # Strategies:
    # (0,0), ("F",1), ("F",2)


def count_expected_play_free_hit_no_unused():
    count = count_expected_outputs(
        2,
        free_transfers=2,
        max_total_hit=None,
        allow_unused_transfers=False,
        next_gw=1,
        max_transfers=2,
        chip_gw_dict={
            1: {"chip_to_play": "free_hit", "chips_allowed": []},
            2: {"chip_to_play": None, "chips_allowed": []},
        },
    )
    assert count == 3
