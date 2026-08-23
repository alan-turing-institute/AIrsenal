"""
Tests for the Strategy result type.

The strategies these replace were dicts keyed by gameweek that went through a
JSON round trip, so an int key became a string key and an int lookup silently
missed. The round-trip tests below are the regression guard for that.
"""

import json

import pytest

from airsenal.core.enums import Chip
from airsenal.optimization.moves import GameweekMove
from airsenal.optimization.strategy import GameweekOutcome, Strategy


def outcome(gameweek, move=None, points=10.0, **kwargs):
    defaults = {
        "discount_factor": 1.0,
        "points_hit": 0,
        "free_transfers": 1,
    }
    return GameweekOutcome(
        gameweek=gameweek,
        move=move if move is not None else GameweekMove(),
        points=points,
        **{**defaults, **kwargs},
    )


def test_empty_strategy():
    strategy = Strategy(root_gameweek=3)
    assert len(strategy) == 0
    assert strategy.total_score == 0
    assert strategy.gameweeks == ()
    assert strategy.label() == ""


def test_extend_does_not_mutate_the_original():
    # Workers hand the same strategy to every child branch, so extending one
    # branch must not be visible to its siblings.
    root = Strategy(root_gameweek=3)
    left = root.extend(outcome(3, GameweekMove(1)))
    right = root.extend(outcome(3, GameweekMove(2)))
    assert len(root) == 0
    assert left.label() == "1"
    assert right.label() == "2"


def test_totals_and_labels():
    strategy = (
        Strategy(root_gameweek=3)
        .extend(outcome(3, GameweekMove(1), points=50.0, points_hit=0))
        .extend(outcome(4, GameweekMove(2), points=40.0, points_hit=4))
        .extend(outcome(5, GameweekMove(chip=Chip.WILDCARD), points=45.0))
    )
    assert strategy.total_score == 135.0
    assert strategy.total_points_hit == 4
    assert strategy.gameweeks == (3, 4, 5)
    assert strategy.label() == "1-2-W"
    assert strategy.chips_played == (None, None, Chip.WILDCARD)


def test_outcome_lookup_is_by_gameweek_number():
    strategy = Strategy(root_gameweek=3).extend(outcome(4, points=7.0))
    assert strategy.outcome(4).points == 7.0
    # gameweek 3 is the root, but no outcome was recorded for it
    with pytest.raises(KeyError, match="nothing for gameweek 3"):
        strategy.outcome(3)


def test_undiscounted_points_backs_out_the_discount():
    got = outcome(4, points=46.5, discount_factor=0.5)
    assert got.undiscounted_points == 93.0


def test_undiscounted_points_survives_a_zero_discount():
    assert outcome(4, points=0.0, discount_factor=0.0).undiscounted_points == 0.0


@pytest.mark.parametrize(
    "move",
    [
        GameweekMove(),
        GameweekMove(2),
        GameweekMove(chip=Chip.WILDCARD),
        GameweekMove(chip=Chip.FREE_HIT),
        GameweekMove(1, Chip.BENCH_BOOST),
        GameweekMove(0, Chip.TRIPLE_CAPTAIN),
    ],
)
def test_round_trip_through_json_shape(move):
    original = Strategy(root_gameweek=3).extend(
        outcome(
            3,
            move,
            points=12.5,
            discount_factor=0.9,
            points_hit=4,
            free_transfers=2,
            players_in=(1, 2),
            players_out=(3, 4),
            bank=15,
        )
    )
    # a real dump goes through json, which turns tuples into lists and would
    # turn any int dict key into a string - hence the explicit dumps/loads
    restored = Strategy.from_dict(json.loads(json.dumps(original.to_dict())))
    assert restored == original
    assert restored.outcome(3).move == move


def test_round_trip_of_a_multi_gameweek_strategy():
    original = (
        Strategy(root_gameweek=10)
        .extend(outcome(10, GameweekMove(1), players_in=(5,), players_out=(6,)))
        .extend(outcome(11, GameweekMove(chip=Chip.FREE_HIT)))
    )
    restored = Strategy.from_dict(json.loads(json.dumps(original.to_dict())))
    assert restored == original
    # gameweeks are list elements, not dict keys, so int lookup keeps working
    assert restored.outcome(11).chip is Chip.FREE_HIT


@pytest.mark.parametrize(
    ("moves", "expected"),
    [
        ([GameweekMove(), GameweekMove()], True),
        ([GameweekMove(), GameweekMove(1)], False),
        ([GameweekMove(chip=Chip.WILDCARD)], False),
        ([GameweekMove(0, Chip.BENCH_BOOST)], False),
        ([], True),
    ],
)
def test_is_baseline(moves, expected):
    strategy = Strategy(root_gameweek=3)
    for i, move in enumerate(moves):
        strategy = strategy.extend(outcome(3 + i, move))
    assert strategy.is_baseline is expected
