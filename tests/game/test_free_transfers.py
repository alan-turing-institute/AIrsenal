"""
How many free transfers an entry has.

The search accrues free transfers with `calc_free_transfers`; the count it starts
from comes from `get_free_transfers`. Both go through
`game.scoring.free_transfers_after`.
"""

import pytest

from airsenal.game.enums import Chip
from airsenal.game.scoring import MAX_FREE_TRANSFERS, free_transfers_after
from airsenal.optimization.moves import GameweekMove, calc_free_transfers


@pytest.mark.parametrize(
    ("n_transfers", "previous", "expected"),
    [
        (0, 1, 2),  # an idle week accrues one
        (0, 4, 5),  # up to the cap
        (0, 5, 5),  # and no further
        (1, 3, 3),  # one transfer spends the one accrued
        (2, 3, 2),  # two spends one of the bank
        (5, 1, 1),  # never below one
    ],
)
def test_free_transfers_accrue_to_the_cap(n_transfers, previous, expected):
    assert free_transfers_after(n_transfers, previous) == expected


def test_idle_weeks_reach_the_documented_maximum():
    """
    Four idle gameweeks from one free transfer reach MAX_FREE_TRANSFERS.

    The estimate used to stop at 2, so the search was told it had 2 and then
    charged a points hit for moves FPL would have given away.
    """
    free_transfers = 1
    for _ in range(4):
        free_transfers = free_transfers_after(0, free_transfers)
    assert free_transfers == MAX_FREE_TRANSFERS


@pytest.mark.parametrize("chip", [Chip.WILDCARD, Chip.FREE_HIT])
def test_rebuilding_the_squad_leaves_the_count_alone(chip):
    # Changed in 24/25: playing a wildcard or free hit no longer resets you to 1.
    assert free_transfers_after(15, 3, rebuilds_squad=True) == 3
    assert calc_free_transfers(GameweekMove(chip=chip), 3) == 3


@pytest.mark.parametrize("max_free_transfers", [2, 5])
@pytest.mark.parametrize("n_transfers", range(6))
@pytest.mark.parametrize("prev_free_transfers", range(6))
def test_the_count_stays_between_one_and_the_cap(
    max_free_transfers, n_transfers, prev_free_transfers
):
    """Whatever the cap, and whatever is spent, the count never leaves the range."""
    got = calc_free_transfers(
        GameweekMove(n_transfers), prev_free_transfers, max_free_transfers
    )
    assert 1 <= got <= max_free_transfers


def test_the_move_shaped_wrapper_agrees_with_the_rule():
    """`calc_free_transfers` is a face on the same arithmetic, not a second copy."""
    for n_transfers in range(4):
        for previous in range(1, MAX_FREE_TRANSFERS + 1):
            assert calc_free_transfers(
                GameweekMove(n_transfers), previous
            ) == free_transfers_after(n_transfers, previous)
