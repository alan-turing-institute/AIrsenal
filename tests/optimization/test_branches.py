"""
Which moves the tree search branches into, and how many plans that adds up to.

`next_week_transfers` answers the first and `count_expected_outputs` the second,
and both live with the search rather than with `GameweekMove`: only a tree walk
asks either question, and the count exists to size a progress bar.
"""

from airsenal.game.enums import Chip
from airsenal.optimization.moves import ChipSchedule, GameweekChips
from airsenal.optimization.transfer_optimizers.tree_search import (
    count_expected_outputs,
    next_week_transfers,
)


def as_labels(
    results: list[tuple],
) -> list[tuple]:
    """
    Replace the GameweekMove in each result with its short label.

    The labels ("W", "T2", "0") are the wire format written to the suggestion
    table, so asserting on them keeps these tests readable and pins the encoding.
    """
    return [(move.label(), *rest) for move, *rest in results]


def test_next_week_transfers_no_chips_no_constraints():
    # First week (blank starting strat with 1 free transfer available)
    free_transfers, hit_so_far = 1, 0
    # No chips or constraints
    actual = as_labels(
        next_week_transfers(
            free_transfers,
            hit_so_far,
            max_total_hit=None,
            allow_unused_transfers=True,
            max_opt_transfers=2,
        )
    )
    # (no. transfers, free transfers next week, total points hit, points hit this gw)
    expected = [("0", 2, 0, 0), ("1", 1, 0, 0), ("2", 1, 4, 4)]
    assert actual == expected


def test_next_week_transfers_no_free_transfers_available():
    # First week (blank starting strat with no free transfer available)
    free_transfers, hit_so_far = 0, 0
    # No chips or constraints
    actual = as_labels(
        next_week_transfers(
            free_transfers,
            hit_so_far,
            max_total_hit=None,
            allow_unused_transfers=True,
            max_opt_transfers=2,
        )
    )
    # (no. transfers, free transfers next week, total points hit, points hit this gw)
    expected = [("0", 1, 0, 0), ("1", 1, 4, 4), ("2", 1, 8, 8)]
    assert actual == expected


def test_next_week_transfers_with_hits_already_taken():
    # First week (blank starting strat with 4 points hits already taken)
    free_transfers, hit_so_far = 1, 4
    # No chips or constraints
    actual = as_labels(
        next_week_transfers(
            free_transfers,
            hit_so_far,
            max_total_hit=None,
            allow_unused_transfers=True,
            max_opt_transfers=2,
        )
    )
    # (no. transfers, free transfers next week, total points hit, points hit this gw)
    expected = [("0", 2, 4, 0), ("1", 1, 4, 0), ("2", 1, 8, 4)]
    assert actual == expected


def test_next_week_transfers_no_chips_no_constraints_max5():
    # First week (blank starting strat with 1 free transfer available)
    free_transfers, hit_so_far = 1, 0
    # No chips or constraints
    actual = as_labels(
        next_week_transfers(
            free_transfers,
            hit_so_far,
            max_total_hit=None,
            allow_unused_transfers=True,
            max_opt_transfers=5,
        )
    )
    # (no. transfers, free transfers next week, total points hit, points hit this gw)
    expected = [
        ("0", 2, 0, 0),
        ("1", 1, 0, 0),
        ("2", 1, 4, 4),
        ("3", 1, 8, 8),
        ("4", 1, 12, 12),
        ("5", 1, 16, 16),
    ]
    assert actual == expected


def test_next_week_transfers_any_chip_no_constraints():
    # All chips, no constraints
    free_transfers, hit_so_far = 1, 0
    actual = as_labels(
        next_week_transfers(
            free_transfers,
            hit_so_far,
            max_total_hit=None,
            max_opt_transfers=2,
            chips=GameweekChips(
                chips_allowed=(
                    Chip.WILDCARD,
                    Chip.FREE_HIT,
                    Chip.BENCH_BOOST,
                    Chip.TRIPLE_CAPTAIN,
                )
            ),
        )
    )
    expected = [
        ("0", 2, 0, 0),
        ("1", 1, 0, 0),
        ("2", 1, 4, 4),
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
    # All chips, no constraints
    free_transfers, hit_so_far = 1, 0
    actual = as_labels(
        next_week_transfers(
            free_transfers,
            hit_so_far,
            max_total_hit=None,
            max_opt_transfers=5,
            chips=GameweekChips(
                chips_allowed=(
                    Chip.WILDCARD,
                    Chip.FREE_HIT,
                    Chip.BENCH_BOOST,
                    Chip.TRIPLE_CAPTAIN,
                )
            ),
        )
    )
    expected = [
        ("0", 2, 0, 0),
        ("1", 1, 0, 0),
        ("2", 1, 4, 4),
        ("3", 1, 8, 8),
        ("4", 1, 12, 12),
        ("5", 1, 16, 16),
        ("W", 1, 0, 0),
        ("F", 1, 0, 0),
        ("B0", 2, 0, 0),
        ("B1", 1, 0, 0),
        ("B2", 1, 4, 4),
        ("B3", 1, 8, 8),
        ("B4", 1, 12, 12),
        ("B5", 1, 16, 16),
        ("T0", 2, 0, 0),
        ("T1", 1, 0, 0),
        ("T2", 1, 4, 4),
        ("T3", 1, 8, 8),
        ("T4", 1, 12, 12),
        ("T5", 1, 16, 16),
    ]
    assert actual == expected


def test_next_week_transfers_no_chips_zero_hit():
    # No points hits
    free_transfers, hit_so_far = 1, 0
    actual = as_labels(
        next_week_transfers(
            free_transfers,
            hit_so_far,
            max_total_hit=0,
            allow_unused_transfers=True,
            max_opt_transfers=2,
        )
    )
    expected = [("0", 2, 0, 0), ("1", 1, 0, 0)]
    assert actual == expected


def test_next_week_transfers_no_chips_zero_hit_max5():
    # No points hits
    free_transfers, hit_so_far = 1, 0
    actual = as_labels(
        next_week_transfers(
            free_transfers,
            hit_so_far,
            max_total_hit=0,
            allow_unused_transfers=True,
            max_opt_transfers=5,
        )
    )
    expected = [("0", 2, 0, 0), ("1", 1, 0, 0)]
    assert actual == expected


def test_next_week_transfers_2ft_no_unused():
    # 2 free transfers available, no wasted transfers
    free_transfers, hit_so_far = 2, 0
    actual = as_labels(
        next_week_transfers(
            free_transfers,
            hit_so_far,
            max_total_hit=None,
            allow_unused_transfers=False,
            max_opt_transfers=2,
            max_free_transfers=2,
        )
    )
    expected = [("1", 2, 0, 0), ("2", 1, 0, 0)]
    assert actual == expected


def test_next_week_transfers_5ft_no_unused_max5():
    # 2 free transfers available, no wasted transfers
    free_transfers, hit_so_far = 5, 0
    actual = as_labels(
        next_week_transfers(
            free_transfers,
            hit_so_far,
            max_total_hit=None,
            allow_unused_transfers=False,
            max_opt_transfers=5,
            max_free_transfers=5,
        )
    )
    expected = [
        ("1", 5, 0, 0),
        ("2", 4, 0, 0),
        ("3", 3, 0, 0),
        ("4", 2, 0, 0),
        ("5", 1, 0, 0),
    ]
    assert actual == expected


def test_next_week_transfers_3ft_no_hit_max5():
    # 2 free transfers available, no wasted transfers
    free_transfers, hit_so_far = 3, 0
    actual = as_labels(
        next_week_transfers(
            free_transfers,
            hit_so_far,
            max_total_hit=0,
            allow_unused_transfers=False,
            max_opt_transfers=5,
            max_free_transfers=5,
        )
    )
    expected = [("0", 4, 0, 0), ("1", 3, 0, 0), ("2", 2, 0, 0), ("3", 1, 0, 0)]
    assert actual == expected


def test_next_week_transfers_chips_already_used():
    # Chips allowed but previously used
    free_transfers, hit_so_far = 1, 0
    chips_played = [
        Chip.WILDCARD,
        Chip.FREE_HIT,
        Chip.BENCH_BOOST,
        Chip.TRIPLE_CAPTAIN,
    ]
    actual = as_labels(
        next_week_transfers(
            free_transfers,
            hit_so_far,
            chips_played,
            max_total_hit=None,
            max_opt_transfers=2,
            # every chip is allowed this week, but all of them are already spent
            chips=GameweekChips(
                chips_allowed=(
                    Chip.WILDCARD,
                    Chip.FREE_HIT,
                    Chip.BENCH_BOOST,
                    Chip.TRIPLE_CAPTAIN,
                )
            ),
        )
    )
    expected = [("0", 2, 0, 0), ("1", 1, 0, 0), ("2", 1, 4, 4)]
    assert actual == expected


def test_next_week_transfers_play_wildcard():
    free_transfers, hit_so_far = 1, 0
    actual = as_labels(
        next_week_transfers(
            free_transfers,
            hit_so_far,
            max_total_hit=None,
            max_opt_transfers=2,
            chips=GameweekChips(Chip.WILDCARD),
        )
    )
    expected = [("W", 1, 0, 0)]
    assert actual == expected


def test_next_week_transfers_2ft_allow_wildcard():
    free_transfers, hit_so_far = 2, 0
    actual = as_labels(
        next_week_transfers(
            free_transfers,
            hit_so_far,
            max_total_hit=None,
            max_opt_transfers=2,
            chips=GameweekChips(chips_allowed=(Chip.WILDCARD,)),
            max_free_transfers=2,
        )
    )
    expected = [("0", 2, 0, 0), ("1", 2, 0, 0), ("2", 1, 0, 0), ("W", 2, 0, 0)]
    assert actual == expected


def test_next_week_transfers_5ft_allow_wildcard():
    free_transfers, hit_so_far = 5, 0
    actual = as_labels(
        next_week_transfers(
            free_transfers,
            hit_so_far,
            max_total_hit=None,
            max_opt_transfers=5,
            chips=GameweekChips(chips_allowed=(Chip.WILDCARD,)),
            max_free_transfers=5,
        )
    )
    expected = [
        ("0", 5, 0, 0),
        ("1", 5, 0, 0),
        ("2", 4, 0, 0),
        ("3", 3, 0, 0),
        ("4", 2, 0, 0),
        ("5", 1, 0, 0),
        ("W", 5, 0, 0),
    ]
    assert actual == expected


def test_next_week_transfers_2ft_allow_wildcard_no_unused():
    free_transfers, hit_so_far = 2, 0
    actual = as_labels(
        next_week_transfers(
            free_transfers,
            hit_so_far,
            max_total_hit=None,
            allow_unused_transfers=False,
            max_opt_transfers=2,
            chips=GameweekChips(chips_allowed=(Chip.WILDCARD,)),
            max_free_transfers=2,
        )
    )
    expected = [("1", 2, 0, 0), ("2", 1, 0, 0), ("W", 2, 0, 0)]
    assert actual == expected


def test_next_week_transfers_2ft_play_wildcard():
    free_transfers, hit_so_far = 2, 0
    actual = as_labels(
        next_week_transfers(
            free_transfers,
            hit_so_far,
            max_total_hit=None,
            max_opt_transfers=2,
            chips=GameweekChips(Chip.WILDCARD),
        )
    )
    expected = [("W", 2, 0, 0)]
    assert actual == expected


def test_next_week_transfers_2ft_play_bench_boost_no_unused():
    free_transfers, hit_so_far = 2, 0
    actual = as_labels(
        next_week_transfers(
            free_transfers,
            hit_so_far,
            max_total_hit=None,
            allow_unused_transfers=False,
            max_opt_transfers=2,
            chips=GameweekChips(Chip.BENCH_BOOST),
            max_free_transfers=2,
        )
    )
    expected = [("B1", 2, 0, 0), ("B2", 1, 0, 0)]
    assert actual == expected


def test_next_week_transfers_play_triple_captain_max_transfers_3():
    free_transfers, hit_so_far = 1, 0
    actual = as_labels(
        next_week_transfers(
            free_transfers,
            hit_so_far,
            max_total_hit=None,
            allow_unused_transfers=True,
            max_opt_transfers=3,
            chips=GameweekChips(Chip.TRIPLE_CAPTAIN),
        )
    )
    expected = [("T0", 2, 0, 0), ("T1", 1, 0, 0), ("T2", 1, 4, 4), ("T3", 1, 8, 8)]
    assert actual == expected


def test_count_expected_outputs_no_chips_no_constraints():
    # No constraints or chips, expect 3**n_gameweeks strategies
    count, _ = count_expected_outputs(
        3,
        free_transfers=1,
        max_total_hit=None,
        allow_unused_transfers=True,
        next_gw=1,
        max_opt_transfers=2,
        chip_schedule=ChipSchedule(),
    )
    assert count == 3**3


def test_count_expected_outputs_no_chips_no_constraints_max5():
    # No constraints or chips, expect 6**n_gameweeks strategies (0 to 5 transfers
    # each week)
    count, _ = count_expected_outputs(
        3,
        free_transfers=1,
        max_total_hit=None,
        allow_unused_transfers=True,
        next_gw=1,
        max_opt_transfers=5,
        chip_schedule=ChipSchedule(),
    )
    assert count == 6**3


def test_count_expected_outputs_no_chips_zero_hit():
    """
    Max hit 0.

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
        chip_schedule=ChipSchedule(),
    )
    assert count == 13


def test_count_expected_outputs_no_chips_zero_hit_max5():
    """
    Max hit 0, max 5 transfers.

    Adds (0, 0, 3) to the strategies of
    test_count_expected_outputs_no_chips_zero_hit above.
    """
    count, _ = count_expected_outputs(
        3,
        free_transfers=1,
        max_total_hit=0,
        next_gw=1,
        max_opt_transfers=5,
        chip_schedule=ChipSchedule(),
    )
    assert count == 14


def test_count_expected_outputs_no_chips_2ft_no_unused():
    """
    Start with 2 free transfers, none allowed to go unused.

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
    Start with 5 free transfers over 2 weeks, none allowed to go unused.

    Include:
    (0, 0),
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
    (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5),
    (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5),
    (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5),
    (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5),
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
    assert count == 30


def test_count_expected_wildcard_allowed_no_constraints():
    """
    Wildcard over 2 weeks, no constraints.

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
        chip_schedule=ChipSchedule.from_weeks([1, 2, 3], {Chip.WILDCARD: 0}),
    )
    assert count == 15


def test_count_expected_bench_boost_allowed_no_constraints():
    """
    Bench boost over 2 weeks, no constraints.

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
        chip_schedule=ChipSchedule.from_weeks([1, 2, 3], {Chip.BENCH_BOOST: 0}),
    )
    assert count == 27


def test_count_expected_play_wildcard_no_constraints():
    """
    Wildcard forced in the first week.

    Strategies: ("W", 0), ("W", 1), ("W", 2).

    None of those is the baseline of no transfers at all, so the baseline is
    excluded from the tree and counted separately - four strategies in total.
    """
    count, baseline_excluded = count_expected_outputs(
        2,
        free_transfers=1,
        max_total_hit=None,
        allow_unused_transfers=True,
        next_gw=1,
        max_opt_transfers=2,
        chip_schedule=ChipSchedule.from_weeks([1, 2], {Chip.WILDCARD: 1}),
    )
    assert (count, baseline_excluded) == (4, True)


def test_count_expected_play_free_hit_no_unused():
    """
    Free hit forced in the first week, 2 free transfers, none unused.

    Strategies: ("F", 1), ("F", 2), plus the separately-counted (0, 0) baseline.
    """
    count, baseline_excluded = count_expected_outputs(
        2,
        free_transfers=2,
        max_total_hit=None,
        allow_unused_transfers=False,
        next_gw=1,
        max_opt_transfers=2,
        chip_schedule=ChipSchedule.from_weeks([1, 2], {Chip.FREE_HIT: 1}),
    )
    assert (count, baseline_excluded) == (4, True)
