"""
The per-gameweek transfer searches, on simplified scenarios.

A dummy squad and mocked predictions, so the answer is known: the best single
transfer, the best pair, and the promise each strategy makes about how many
candidate squads it will consider.
"""

from operator import itemgetter
from unittest import mock

from airsenal.core.console import console
from airsenal.optimization.moves import GameweekMove
from airsenal.optimization.protocols import TransferRequest
from airsenal.optimization.strategies import DEFAULT_STRATEGIES
from airsenal.optimization.strategies.double import make_optimum_double_transfer
from airsenal.optimization.strategies.single import make_optimum_single_transfer
from airsenal.squad.squad import Squad


class DummyPlayer:
    """A fake player that can be added to a squad, with a chosen expected score."""

    def __init__(self, player_id, position, points_dict):
        """
        Each player gets its own club and a price of zero.

        So a test squad can hold any number of them without tripping the
        three-per-club limit or the budget.
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
    Fill a squad with dummy players.

    `player_points_dict` is {player_id: {gameweek: points}}.
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
    Build a stand-in for `get_predicted_points`.

    `point_dict` is keyed by position: {"GK": {player_id: points}, "DEF": {}, ...}.
    """

    def mock_get_predicted_points(
        gameweeks, tag, position, team=None, season=None, dbsession=None
    ):
        """
        Ordered exactly as the real `get_predicted_points` orders its result.

        Dummy players rather than bare ids, so `Squad.add_player` can take them.
        """
        output_pid_list = [(k, v) for k, v in point_dict[position].items()]
        output_pid_list.sort(key=itemgetter(1), reverse=True)
        gameweek = next(iter(gameweeks))
        return [
            (DummyPlayer(entry[0], position, {gameweek: entry[1]}), entry[1])
            for entry in output_pid_list
        ]

    return mock_get_predicted_points


def test_subs():
    """The starting eleven is picked correctly when some players are predicted zero."""
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
    """The best available transfer is chosen when every current player scores alike."""
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
        "airsenal.optimization.strategies.single.get_predicted_points",
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
    """A squad with two weak players transfers both of them out."""
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
        "airsenal.optimization.strategies.double.get_predicted_points",
        side_effect=mock_pred_points,
    ):
        new_squad, _pid_out, pid_in = make_optimum_double_transfer(t, "DUMMY", [1])
        # we should expect 201 and 115 to be transferred in, and 1,15 to
        # be transferred out.   115 should be captain
        assert 201 in pid_in
        assert 115 in pid_in
        console.print(new_squad)
        for p in new_squad.players:
            if p.player_id == 115:
                assert p.is_captain is True
            else:
                assert p.is_captain is False


def test_the_progress_steps_counted_match_the_number_promised():
    """A worker's bar is sized by `num_increments`, so the search must hit it.

    The total is the number of candidate squads the strategy says it will
    consider, and it is advanced once per candidate. If the two disagree the bar
    stalls short of the end, or finishes while the search is still going.
    """
    squad = generate_dummy_squad()
    points = {
        position: dict.fromkeys(ids, 2)
        for position, ids in {
            "GK": [0, 1, 100, 101],
            "DEF": [2, 3, 4, 5, 6, 103, 104, 105],
            "MID": [7, 8, 9, 10, 11, 108, 109, 110],
            "FWD": [12, 13, 14, 113, 114],
        }.items()
    }

    # every strategy that reports progress at all: the whole-squad rebuild a
    # wildcard does is the genetic algorithm, which reports nothing back.
    for move in (GameweekMove(1), GameweekMove(2), GameweekMove(3)):
        strategy = DEFAULT_STRATEGIES.create(move)
        steps = 0

        def count_step() -> None:
            nonlocal steps
            steps += 1

        request = TransferRequest(
            move=move,
            squad=squad,
            tag="DUMMY",
            gameweeks=[1],
            root_gw=1,
            season="2526",
            num_iterations=7,
            progress=count_step,
        )
        with mock.patch(
            f"{type(strategy).__module__}.get_predicted_points",
            side_effect=predicted_point_mock_generator(points),
        ):
            strategy.propose(request)

        assert steps == strategy.num_increments(request), move
