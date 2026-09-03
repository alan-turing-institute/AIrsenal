"""
Tests for GameweekMove and the chip schedule.

The compact string encoding of a move is parsed in exactly one place, so it is
pinned here once. How a move changes the free transfer count is in
tests/game/test_free_transfers.py, with the rule it defers to.
"""

import pytest

from airsenal.game.enums import Chip
from airsenal.optimization.moves import (
    ChipSchedule,
    GameweekChips,
    GameweekMove,
    calc_points_hit,
)

LABELS = [
    (GameweekMove(), "0"),
    (GameweekMove(1), "1"),
    (GameweekMove(2), "2"),
    (GameweekMove(15), "15"),
    (GameweekMove(chip=Chip.WILDCARD), "W"),
    (GameweekMove(chip=Chip.FREE_HIT), "F"),
    (GameweekMove(0, Chip.TRIPLE_CAPTAIN), "T0"),
    (GameweekMove(2, Chip.TRIPLE_CAPTAIN), "T2"),
    (GameweekMove(0, Chip.BENCH_BOOST), "B0"),
    (GameweekMove(5, Chip.BENCH_BOOST), "B5"),
]


@pytest.mark.parametrize(("move", "label"), LABELS)
def test_label(move, label):
    assert move.label() == label
    assert str(move) == label


@pytest.mark.parametrize(("move", "label"), LABELS)
def test_parse_round_trip(move, label):
    assert GameweekMove.parse(label) == move


def test_parse_accepts_an_int():
    assert GameweekMove.parse(3) == GameweekMove(3)


def test_parse_rejects_nonsense():
    with pytest.raises(ValueError, match="Unrecognised move label"):
        GameweekMove.parse("X1")


def test_negative_transfers_rejected():
    with pytest.raises(ValueError, match="must not be negative"):
        GameweekMove(-1)


def test_transfers_alongside_a_squad_chip_rejected():
    # A wildcard replaces the squad, so "wildcard plus two transfers" is not a
    # thing the game allows.
    with pytest.raises(ValueError, match="replaces the whole squad"):
        GameweekMove(2, Chip.WILDCARD)


@pytest.mark.parametrize(
    ("move", "rebuilds", "n_in", "carry_forward"),
    [
        (GameweekMove(2), False, 2, True),
        (GameweekMove(chip=Chip.WILDCARD), True, 15, True),
        (GameweekMove(chip=Chip.FREE_HIT), True, 15, False),
        (GameweekMove(1, Chip.BENCH_BOOST), False, 1, True),
        (GameweekMove(1, Chip.TRIPLE_CAPTAIN), False, 1, True),
    ],
)
def test_move_properties(move, rebuilds, n_in, carry_forward):
    assert move.rebuilds_squad is rebuilds
    assert move.n_players_in == n_in
    assert move.carry_forward is carry_forward


@pytest.mark.parametrize("n_transfers", range(5))
@pytest.mark.parametrize("free_transfers", range(6))
@pytest.mark.parametrize(
    "chip", [None, Chip.BENCH_BOOST, Chip.TRIPLE_CAPTAIN, Chip.WILDCARD, Chip.FREE_HIT]
)
def test_calc_points_hit(n_transfers, free_transfers, chip):
    if chip is not None and chip.rebuilds_squad:
        assert calc_points_hit(GameweekMove(chip=chip), free_transfers) == 0
        return
    move = GameweekMove(n_transfers, chip)
    assert calc_points_hit(move, free_transfers) == max(
        0, 4 * (n_transfers - free_transfers)
    )


def test_gameweek_chips_rejects_allowing_and_forcing_at_once():
    with pytest.raises(ValueError, match="Cannot allow"):
        GameweekChips(Chip.WILDCARD, (Chip.FREE_HIT,))


def test_gameweek_chips_allows_only_unplayed_chips():
    chips = GameweekChips(chips_allowed=(Chip.WILDCARD, Chip.FREE_HIT))
    assert chips.allows(Chip.WILDCARD, [None, Chip.FREE_HIT]) is True
    assert chips.allows(Chip.FREE_HIT, [None, Chip.FREE_HIT]) is False
    assert chips.allows(Chip.BENCH_BOOST, []) is False


def test_chip_schedule_from_weeks():
    schedule = ChipSchedule.from_weeks(
        [1, 2, 3],
        {
            Chip.WILDCARD: 0,  # any week
            Chip.FREE_HIT: -1,  # never
            Chip.BENCH_BOOST: 2,  # definitely gameweek 2
            Chip.TRIPLE_CAPTAIN: 9,  # outside the range, so ignored
        },
    )
    assert schedule.for_gameweek(1) == GameweekChips(chips_allowed=(Chip.WILDCARD,))
    # a definite chip displaces the optional ones for that gameweek
    assert schedule.for_gameweek(2) == GameweekChips(Chip.BENCH_BOOST)
    assert schedule.for_gameweek(3) == GameweekChips(chips_allowed=(Chip.WILDCARD,))


def test_chip_schedule_unknown_gameweek_allows_nothing():
    schedule = ChipSchedule.from_weeks([1], {Chip.WILDCARD: 0})
    assert schedule.for_gameweek(7) == GameweekChips()


def test_chip_schedule_rejects_two_chips_in_one_week():
    with pytest.raises(ValueError, match="same week"):
        ChipSchedule.from_weeks([1, 2], {Chip.WILDCARD: 1, Chip.FREE_HIT: 1})


def test_chip_schedule_accepts_chip_names_as_strings():
    # This is the shape the CLI hands over.
    schedule = ChipSchedule.from_weeks([1, 2], {"wildcard": 2, "free_hit": -1})
    assert schedule.for_gameweek(2) == GameweekChips(Chip.WILDCARD)
