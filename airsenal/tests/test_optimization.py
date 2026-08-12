"""
Test the optimization of transfers, generating a few simplified scenarios
and checking that the optimizer finds the expected outcome.
"""

import random
from contextlib import contextmanager
from operator import itemgetter
from unittest import mock
from unittest.mock import Mock, patch

import numpy as np
import pytest

from airsenal.framework.optimization_transfers import (
    make_optimum_double_transfer,
    make_optimum_single_transfer,
    make_optimum_transfers_ga,
)
from airsenal.framework.optimization_utils import (
    count_expected_outputs,
    get_discount_factor,
    next_week_transfers,
)
from airsenal.framework.season import CURRENT_SEASON
from airsenal.framework.squad import Squad

pytestmark = pytest.mark.filterwarnings("ignore:Using purchase price as sale price")


class DummyPlayer:
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
        self.display_name = f"Dummy Player_{player_id}"
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


def _mock_player(player_id, position, team, price, points):
    """
    A player double satisfying CandidatePlayer's needs (used as a `list_players()`
    candidate and, via player_id, for building the "current" squad passed to
    make_optimum_transfers_ga), with a fixed points dict keyed by gameweek.
    """
    player = Mock()
    player.player_id = player_id
    player.name = f"Player {player_id}"
    player.display_name = None
    player.position = lambda _season=None, _pos=position: _pos
    player.team = lambda _season=None, _gameweek=None, _team=team: _team
    player.price = lambda _season=None, _gameweek=None, _price=price: _price
    player.points = points
    return player


@contextmanager
def _mock_player_db(players):
    """
    Make `make_optimum_transfers_ga`/`SquadOpt` work against a fixed in-memory list
    of `_mock_player` doubles instead of the real player database: `list_players`
    supplies the GA's candidate pool, and `get_player`/`get_predicted_points_for_player`
    (called internally whenever Squad.add_player/CandidatePlayer resolve a bare
    player_id) are patched to resolve against the same list.
    """
    lookup = {p.player_id: p for p in players}

    def list_players_side_effect(position=None, **kwargs):
        return [p for p in players if p.position() == position]

    def get_player_side_effect(player_id, dbsession=None):
        return lookup.get(player_id)

    def points_side_effect(player, tag, season=None, dbsession=None):
        pid = player.player_id if hasattr(player, "player_id") else player
        return lookup[pid].points if pid in lookup else {}

    with (
        patch(
            "airsenal.framework.optimization_squad.list_players",
            side_effect=list_players_side_effect,
        ),
        patch(
            "airsenal.framework.optimization_squad.get_predicted_points_for_player",
            side_effect=points_side_effect,
        ),
        patch(
            "airsenal.framework.player.get_player",
            side_effect=get_player_side_effect,
        ),
        patch(
            "airsenal.framework.player.get_predicted_points_for_player",
            side_effect=points_side_effect,
        ),
    ):
        yield


def _build_squad(players, price=40, budget=1000):
    """Build a Squad containing exactly `players` (via _mock_player_db)."""
    squad = Squad(budget=budget, season=CURRENT_SEASON)
    for p in players:
        assert squad.add_player(p.player_id, price=price, gameweek=1)
    return squad


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
        new_squad, _pid_out, pid_in = make_optimum_single_transfer(t, "DUMMY", [1])
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
        new_squad, _pid_out, pid_in = make_optimum_double_transfer(t, "DUMMY", [1])
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


def _make_test_squad_players(weak_ids=()):
    """
    15 players (2 GK, 5 DEF, 5 MID, 3 FWD), each scoring 2 points, except
    `weak_ids` which score 1 - a clear transfer-out candidate.
    """
    positions = ["GK"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    return [
        _mock_player(
            i + 1, pos, f"Team{i + 1}", 40, {1: 1 if (i + 1) in weak_ids else 2}
        )
        for i, pos in enumerate(positions)
    ]


def test_make_optimum_transfers_ga_respects_transfer_cap():
    """
    Even when more than `num_transfers` clear improvements are available, the GA
    must never propose more transfers than the cap allows - this is the behaviour
    that motivated adding a GA-based search for transfer counts above 2, where the
    old exhaustive/random search couldn't scale to the higher counts allowed by
    saving up free transfers. 3 is the smallest count that actually reaches
    make_optimum_transfers_ga via make_best_transfers (1 and 2 use the exhaustive
    make_optimum_single_transfer/make_optimum_double_transfer instead).
    """
    random.seed(1)
    np.random.seed(1)
    tag = "DUMMY"

    # four clear improvements available (one per position), but only 3 transfers
    # allowed - the GA must pick at most 3, never all 4.
    squad_players = _make_test_squad_players(weak_ids={1, 7, 12, 15})
    candidates = [
        _mock_player(102, "GK", "TeamW", 40, {1: 9}),
        _mock_player(103, "DEF", "TeamX", 40, {1: 9}),
        _mock_player(104, "MID", "TeamY", 40, {1: 9}),
        _mock_player(101, "FWD", "TeamZ", 40, {1: 9}),
    ]

    with _mock_player_db([*squad_players, *candidates]):
        squad = _build_squad(squad_players)
        baseline_score = squad.get_expected_points(1, tag)
        new_squad, players_out, players_in = make_optimum_transfers_ga(
            squad, tag, 3, gameweek_range=[1], num_iter=60
        )
        new_score = new_squad.get_expected_points(1, tag)

    assert len(players_out) <= 3
    assert len(players_in) <= 3
    assert len(players_out) == len(players_in)
    # the GA should have found and used at least one of the available improvements
    assert set(players_in) & {102, 103, 104, 101}
    assert new_score >= baseline_score


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
        max_opt_transfers=2,
    )
    # (no. transfers, free transfers next week, total points hit, points hit this gw)
    expected = [(0, 2, 0, 0), (1, 1, 0, 0), (2, 1, 4, 4)]
    assert actual == expected


def test_next_week_transfers_no_free_transfers_available():
    # First week (blank starting strat with no free transfer available)
    strat = (0, 0, {"players_in": {}, "chips_played": {}})
    # No chips or constraints
    actual = next_week_transfers(
        strat,
        max_total_hit=None,
        allow_unused_transfers=True,
        max_opt_transfers=2,
    )
    # (no. transfers, free transfers next week, total points hit, points hit this gw)
    expected = [(0, 1, 0, 0), (1, 1, 4, 4), (2, 1, 8, 8)]
    assert actual == expected


def test_next_week_transfers_with_hits_already_taken():
    # First week (blank starting strat with 4 points hits already taken)
    strat = (1, 4, {"players_in": {}, "chips_played": {}})
    # No chips or constraints
    actual = next_week_transfers(
        strat,
        max_total_hit=None,
        allow_unused_transfers=True,
        max_opt_transfers=2,
    )
    # (no. transfers, free transfers next week, total points hit, points hit this gw)
    expected = [(0, 2, 4, 0), (1, 1, 4, 0), (2, 1, 8, 4)]
    assert actual == expected


def test_next_week_transfers_no_chips_no_constraints_max5():
    # First week (blank starting strat with 1 free transfer available). Even with
    # max_opt_transfers=5, the coarse candidate set only offers "use all available
    # free transfers" (here, 1) in addition to 0/1/2, so with only 1 free transfer
    # available this is identical to the max_opt_transfers=2 case.
    strat = (1, 0, {"players_in": {}, "chips_played": {}})
    # No chips or constraints
    actual = next_week_transfers(
        strat,
        max_total_hit=None,
        allow_unused_transfers=True,
        max_opt_transfers=5,
    )
    # (no. transfers, free transfers next week, total points hit, points hit this gw)
    expected = [(0, 2, 0, 0), (1, 1, 0, 0), (2, 1, 4, 4)]
    assert actual == expected


def test_next_week_transfers_any_chip_no_constraints():
    # All chips, no constraints
    strat = (1, 0, {"players_in": {}, "chips_played": {}})
    actual = next_week_transfers(
        strat,
        max_total_hit=None,
        max_opt_transfers=2,
        chips={
            "chips_allowed": ["wildcard", "free_hit", "bench_boost", "triple_captain"],
            "chip_to_play": None,
        },
    )
    expected = [
        (0, 2, 0, 0),
        (1, 1, 0, 0),
        (2, 1, 4, 4),
        ("W", 1, 0, 0),
        ("F", 1, 0, 0),
        ("B0", 2, 0, 0),
        ("B1", 1, 0, 0),
        ("B2", 1, 4, 4),
        ("T0", 2, 0, 0),
        ("T1", 1, 0, 0),
        ("T2", 1, 4, 4),
    ]
    assert actual == expected


def test_next_week_transfers_any_chip_no_constraints_max5():
    # All chips, no constraints. With only 1 free transfer available, the coarse
    # candidate set collapses to the same as max_opt_transfers=2 (see
    # test_next_week_transfers_no_chips_no_constraints_max5).
    strat = (1, 0, {"players_in": {}, "chips_played": {}})
    actual = next_week_transfers(
        strat,
        max_total_hit=None,
        max_opt_transfers=5,
        chips={
            "chips_allowed": ["wildcard", "free_hit", "bench_boost", "triple_captain"],
            "chip_to_play": None,
        },
    )
    expected = [
        (0, 2, 0, 0),
        (1, 1, 0, 0),
        (2, 1, 4, 4),
        ("W", 1, 0, 0),
        ("F", 1, 0, 0),
        ("B0", 2, 0, 0),
        ("B1", 1, 0, 0),
        ("B2", 1, 4, 4),
        ("T0", 2, 0, 0),
        ("T1", 1, 0, 0),
        ("T2", 1, 4, 4),
    ]
    assert actual == expected


def test_next_week_transfers_no_chips_zero_hit():
    # No points hits
    strat = (1, 0, {"players_in": {}, "chips_played": {}})
    actual = next_week_transfers(
        strat,
        max_total_hit=0,
        allow_unused_transfers=True,
        max_opt_transfers=2,
    )
    expected = [(0, 2, 0, 0), (1, 1, 0, 0)]
    assert actual == expected


def test_next_week_transfers_no_chips_zero_hit_max5():
    # No points hits
    strat = (1, 0, {"players_in": {}, "chips_played": {}})
    actual = next_week_transfers(
        strat,
        max_total_hit=0,
        allow_unused_transfers=True,
        max_opt_transfers=5,
    )
    expected = [(0, 2, 0, 0), (1, 1, 0, 0)]
    assert actual == expected


def test_next_week_transfers_2ft_no_unused():
    # 2 free transfers available, no wasted transfers
    strat = (2, 0, {"players_in": {}, "chips_played": {}})
    actual = next_week_transfers(
        strat,
        max_total_hit=None,
        allow_unused_transfers=False,
        max_opt_transfers=2,
        max_free_transfers=2,
    )
    expected = [(1, 2, 0, 0), (2, 1, 0, 0)]
    assert actual == expected


def test_next_week_transfers_5ft_no_unused_max5():
    # 5 free transfers available, no wasted transfers. The coarse candidate set
    # offers 1, 2, and "use all 5 available" - not every intermediate value - so the
    # tree can compare a small transfer against spending everything currently banked,
    # without having to separately search 3 and 4 too.
    strat = (5, 0, {"players_in": {}, "chips_played": {}})
    actual = next_week_transfers(
        strat,
        max_total_hit=None,
        allow_unused_transfers=False,
        max_opt_transfers=5,
        max_free_transfers=5,
    )
    expected = [(1, 5, 0, 0), (2, 4, 0, 0), (5, 1, 0, 0)]
    assert actual == expected


def test_next_week_transfers_3ft_no_hit_max5():
    # 2 free transfers available, no wasted transfers
    strat = (3, 0, {"players_in": {}, "chips_played": {}})
    actual = next_week_transfers(
        strat,
        max_total_hit=0,
        allow_unused_transfers=False,
        max_opt_transfers=5,
        max_free_transfers=5,
    )
    expected = [(0, 4, 0, 0), (1, 3, 0, 0), (2, 2, 0, 0), (3, 1, 0, 0)]
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
        max_opt_transfers=2,
    )
    expected = [(0, 2, 0, 0), (1, 1, 0, 0), (2, 1, 4, 4)]
    assert actual == expected


def test_next_week_transfers_play_wildcard():
    strat = (1, 0, {"players_in": {}, "chips_played": {}})
    actual = next_week_transfers(
        strat,
        max_total_hit=None,
        max_opt_transfers=2,
        chips={"chips_allowed": [], "chip_to_play": "wildcard"},
    )
    expected = [("W", 1, 0, 0)]
    assert actual == expected


def test_next_week_transfers_2ft_allow_wildcard():
    strat = (2, 0, {"players_in": {}, "chips_played": {}})
    actual = next_week_transfers(
        strat,
        max_total_hit=None,
        max_opt_transfers=2,
        chips={"chips_allowed": ["wildcard"], "chip_to_play": None},
        max_free_transfers=2,
    )
    expected = [(0, 2, 0, 0), (1, 2, 0, 0), (2, 1, 0, 0), ("W", 2, 0, 0)]
    assert actual == expected


def test_next_week_transfers_5ft_allow_wildcard():
    # Coarse candidate set: 0, 1, 2, and "use all 5 available" - not every
    # intermediate value up to 5.
    strat = (5, 0, {"players_in": {}, "chips_played": {}})
    actual = next_week_transfers(
        strat,
        max_total_hit=None,
        max_opt_transfers=5,
        chips={"chips_allowed": ["wildcard"], "chip_to_play": None},
        max_free_transfers=5,
    )
    expected = [
        (0, 5, 0, 0),
        (1, 5, 0, 0),
        (2, 4, 0, 0),
        (5, 1, 0, 0),
        ("W", 5, 0, 0),
    ]
    assert actual == expected


def test_next_week_transfers_2ft_allow_wildcard_no_unused():
    strat = (2, 0, {"players_in": {}, "chips_played": {}})
    actual = next_week_transfers(
        strat,
        max_total_hit=None,
        allow_unused_transfers=False,
        max_opt_transfers=2,
        chips={"chips_allowed": ["wildcard"], "chip_to_play": None},
        max_free_transfers=2,
    )
    expected = [(1, 2, 0, 0), (2, 1, 0, 0), ("W", 2, 0, 0)]
    assert actual == expected


def test_next_week_transfers_2ft_play_wildcard():
    strat = (2, 0, {"players_in": {}, "chips_played": {}})
    actual = next_week_transfers(
        strat,
        max_total_hit=None,
        max_opt_transfers=2,
        chips={"chips_allowed": [], "chip_to_play": "wildcard"},
    )
    expected = [("W", 2, 0, 0)]
    assert actual == expected


def test_next_week_transfers_2ft_play_bench_boost_no_unused():
    strat = (2, 0, {"players_in": {}, "chips_played": {}})
    actual = next_week_transfers(
        strat,
        max_total_hit=None,
        allow_unused_transfers=False,
        max_opt_transfers=2,
        chips={"chips_allowed": [], "chip_to_play": "bench_boost"},
        max_free_transfers=2,
    )
    expected = [("B1", 2, 0, 0), ("B2", 1, 0, 0)]
    assert actual == expected


def test_next_week_transfers_play_triple_captain_max_transfers_3():
    # With 1 free transfer available, "use all available" coincides with 1, so the
    # coarse candidate set here is just {0, 1, 2} - same as max_opt_transfers=2.
    strat = (1, 0, {"players_in": {}, "chips_played": {}})
    actual = next_week_transfers(
        strat,
        max_total_hit=None,
        allow_unused_transfers=True,
        max_opt_transfers=3,
        chips={"chips_allowed": [], "chip_to_play": "triple_captain"},
    )
    expected = [("T0", 2, 0, 0), ("T1", 1, 0, 0), ("T2", 1, 4, 4)]
    assert actual == expected


def test_count_expected_outputs_no_chips_no_constraints():
    # No constraints or chips, expect 3**num_gameweeks strategies
    count, _ = count_expected_outputs(
        3,
        free_transfers=1,
        max_total_hit=None,
        allow_unused_transfers=True,
        next_gw=1,
        max_opt_transfers=2,
        chip_gw_dict={},
    )
    assert count == 3**3


def test_count_expected_outputs_no_chips_no_constraints_max5():
    # No constraints or chips. With max_opt_transfers=5, each week now branches on
    # the coarse {0, 1, 2, "use all available free transfers"} set rather than every
    # integer 0-5 - starting from 1 free transfer, that's 3 options (0/1/2) per week
    # unless accumulated free transfers exceed 2, at which point "use all available"
    # becomes a distinct 4th option. Verified directly against next_week_transfers:
    # 27 (3**3, as if max_opt_transfers were 2) plus 1 extra leaf where saving up
    # transfers for 2 weeks (0, 0, ...) unlocks a genuine 4th, 3-free-transfer option
    # in the final week.
    count, _ = count_expected_outputs(
        3,
        free_transfers=1,
        max_total_hit=None,
        allow_unused_transfers=True,
        next_gw=1,
        max_opt_transfers=5,
        chip_gw_dict={},
    )
    assert count == 28


def test_count_expected_outputs_no_chips_zero_hit():
    """
    Max hit 0
    Include:
    (0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0), (0, 1, 1), (0, 1, 2),
    (0, 2, 0), (0, 2, 1), (1, 0, 0), (1, 0, 1), (1, 0, 2), (1, 1, 0), (1, 1, 1)
    Exclude:
    (0, 2, 2), (1, 1, 2), (1, 2, 0), (1, 2, 1), (1, 2, 2), (2, 0, 0), (2, 0, 1),
    (2, 0, 2), (2, 1, 0), (2, 1, 1), (2, 1, 2), (2, 2, 0), (2, 2, 1), (2, 2, 2)
    """
    count, _ = count_expected_outputs(
        3,
        free_transfers=1,
        max_total_hit=0,
        next_gw=1,
        max_opt_transfers=2,
        chip_gw_dict={},
    )
    assert count == 13


def test_count_expected_outputs_no_chips_zero_hit_max5():
    """
    Max hit 0
    Max 5 transfers
    Adds (0, 0, 3) to valid strategies compared to
    test_count_expected_outputs_no_chips_zero_hit above
    """
    count, _ = count_expected_outputs(
        3,
        free_transfers=1,
        max_total_hit=0,
        next_gw=1,
        max_opt_transfers=5,
        chip_gw_dict={},
    )
    assert count == 14


def test_count_expected_outputs_no_chips_2ft_no_unused():
    """
    Start with 2 FT and no unused
    Include:
    (0, 0, 0), (1, 1, 1), (1, 1, 2), (1, 2, 0), (1, 2, 1), (1, 2, 2), (2, 0, 1),
    (2, 0, 2), (2, 1, 0), (2, 1, 1), (2, 1, 2), (2, 2, 0), (2, 2, 1), (2, 2, 2)
    Exclude:
    (0, 0, 1), (0, 0, 2), (0, 1, 0), (0, 1, 1), (0, 1, 2), (0, 2, 0), (0, 2, 1),
    (0, 2, 2), (1, 0, 0), (1, 0, 1), (1, 0, 2), (1, 1, 0), (2, 0, 0)
    """
    count, _ = count_expected_outputs(
        3,
        free_transfers=2,
        max_total_hit=None,
        allow_unused_transfers=False,
        next_gw=1,
        max_opt_transfers=2,
        max_free_transfers=2,
    )
    assert count == 14


def test_count_expected_outputs_no_chips_5ft_no_unused_max5():
    """
    Start with 5 FT and no unused over 2 weeks. With the coarse candidate set, week 1
    only offers {1, 2, "use all 5 available"} (0 is excluded by allow_unused=False
    forcing at least 1 transfer while at the free-transfer cap), each leading to a
    different week-2 free-transfer count and hence a different (also coarse) week-2
    candidate set - verified directly against next_week_transfers:
    Include:
    (1, 1), (1, 2), (1, 5),
    (2, 0), (2, 1), (2, 2), (2, 4),
    (5, 0), (5, 1), (5, 2),
    plus the baseline (0, 0) strategy, excluded from the tree by allow_unused=False
    (since it would waste transfers while already at the 5-transfer cap) and added
    back separately by count_expected_outputs.
    """
    count, _ = count_expected_outputs(
        2,
        free_transfers=5,
        max_total_hit=None,
        allow_unused_transfers=False,
        next_gw=1,
        max_opt_transfers=5,
        max_free_transfers=5,
    )
    assert count == 11


def test_count_expected_wildcard_allowed_no_constraints():
    """
    Wildcard, 2 weeks, no constraints
    Strategies:
    (0, 0), (0, 1), (0, 2), (0, 'W'), (1, 0), (1, 1), (1, 2), (1, 'W'), (2, 0),
    (2, 1), (2, 2), (2, 'W'), ('W', 0), ('W', 1), ('W', 2)
    """
    count, _ = count_expected_outputs(
        2,
        free_transfers=1,
        max_total_hit=None,
        allow_unused_transfers=True,
        next_gw=1,
        max_opt_transfers=2,
        chip_gw_dict={
            1: {"chips_allowed": ["wildcard"]},
            2: {"chips_allowed": ["wildcard"]},
            3: {"chips_allowed": ["wildcard"]},
        },
    )
    assert count == 15


def count_expected_bench_boost_allowed_no_constraints():
    """
    Bench boost, 2 weeks, no constraints
    Strategies:
    (0, 0), (0, 1), (0, 2), (0, 'B0'), (0, 'B1'), (0, 'B2'), (1, 0), (1, 1), (1, 2),
    (1, 'B0'), (1, 'B1'), (1, 'B2'), (2, 0), (2, 1), (2, 2), (2, 'B0'), (2, 'B1'),
    (2, 'B2'), ('B0', 0), ('B0', 1), ('B0', 2), ('B1', 0), ('B1', 1), ('B1', 2),
    ('B2', 0), ('B2', 1), ('B2', 2),
    """
    count, _ = count_expected_outputs(
        2,
        free_transfers=1,
        max_total_hit=None,
        allow_unused_transfers=True,
        next_gw=1,
        max_opt_transfers=2,
        chip_gw_dict={
            1: {"chips_allowed": ["bench_boost"]},
            2: {"chips_allowed": ["bench_boost"]},
            3: {"chips_allowed": ["bench_boost"]},
        },
    )
    assert count == 27


def count_expected_play_wildcard_no_constraints():
    """
    Force playing wildcard in first week
    Strategies:
    ("W",0), ("W,1), ("W",2)
    """
    count, _ = count_expected_outputs(
        2,
        free_transfers=1,
        max_total_hit=None,
        allow_unused_transfers=True,
        next_gw=1,
        max_opt_transfers=2,
        chip_gw_dict={
            1: {"chip_to_play": "wildcard", "chips_allowed": []},
            2: {"chip_to_play": None, "chips_allowed": []},
        },
    )
    assert count == 3


def count_expected_play_free_hit_no_unused():
    """
    Force playing free hit in first week, 2FT, don't allow unused
    Strategies:
    (0,0), ("F",1), ("F",2)
    """
    count, _ = count_expected_outputs(
        2,
        free_transfers=2,
        max_total_hit=None,
        allow_unused_transfers=False,
        next_gw=1,
        max_opt_transfers=2,
        chip_gw_dict={
            1: {"chip_to_play": "free_hit", "chips_allowed": []},
            2: {"chip_to_play": None, "chips_allowed": []},
        },
    )
    assert count == 3
