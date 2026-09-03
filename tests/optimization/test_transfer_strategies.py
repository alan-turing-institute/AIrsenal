"""
Tests for the transfer strategy table and the move-to-strategy mapping.

These check the wiring, not the searches themselves - the searches are covered
by tests/optimization/test_strategy_searches.py.
"""

import pytest

from airsenal.core.lookup import ConfigError
from airsenal.game.enums import Chip
from airsenal.optimization.moves import GameweekMove
from airsenal.optimization.protocols import (
    Proposal,
    TransferRequest,
    strategy_total,
)
from airsenal.optimization.strategies import (
    DEFAULT_STRATEGIES,
    TRANSFER_STRATEGIES,
    StrategySet,
)


def test_all_five_strategies_are_registered():
    assert sorted(TRANSFER_STRATEGIES) == [
        "double",
        "full_squad",
        "none",
        "random",
        "single",
    ]


@pytest.mark.parametrize("name", sorted(TRANSFER_STRATEGIES))
def test_every_strategy_can_size_its_own_progress_bar(name):
    """
    `num_increments` is beyond what the protocol requires, so it is checked here.

    `propose` is not: tests/test_component_tables.py asserts the protocol method
    of every entry of all five tables, this one included.
    """
    assert callable(TRANSFER_STRATEGIES[name]().num_increments)


def test_unknown_strategy_lists_the_valid_ones():
    with pytest.raises(ConfigError, match="Choose from: double, full_squad"):
        StrategySet(rebuild="wildcard-ish").create(GameweekMove(chip=Chip.WILDCARD))


@pytest.mark.parametrize(
    ("move", "expected"),
    [
        (GameweekMove(0), "none"),
        (GameweekMove(1), "single"),
        (GameweekMove(2), "double"),
        (GameweekMove(3), "random"),
        (GameweekMove(15), "random"),
        (GameweekMove(chip=Chip.WILDCARD), "full_squad"),
        (GameweekMove(chip=Chip.FREE_HIT), "full_squad"),
        (GameweekMove(0, Chip.BENCH_BOOST), "none"),
        (GameweekMove(1, Chip.BENCH_BOOST), "single"),
        (GameweekMove(2, Chip.TRIPLE_CAPTAIN), "double"),
        (GameweekMove(4, Chip.TRIPLE_CAPTAIN), "random"),
    ],
)
def test_strategy_name_for(move, expected):
    assert DEFAULT_STRATEGIES.name_for(move) == expected
    assert DEFAULT_STRATEGIES.create(move) is not None


@pytest.mark.parametrize("n_transfers", range(16))
@pytest.mark.parametrize(
    "chip", [None, Chip.BENCH_BOOST, Chip.TRIPLE_CAPTAIN, Chip.WILDCARD, Chip.FREE_HIT]
)
def test_every_move_maps_to_a_registered_strategy(n_transfers, chip):
    if chip is not None and chip.rebuilds_squad:
        move = GameweekMove(chip=chip)
    else:
        move = GameweekMove(n_transfers, chip)
    assert DEFAULT_STRATEGIES.name_for(move) in TRANSFER_STRATEGIES


def test_none_strategy_keeps_the_squad_and_advances_the_progress_bar():
    steps = 0

    def count_step() -> None:
        nonlocal steps
        steps += 1

    squad = object()
    request = TransferRequest(
        move=GameweekMove(0),
        squad=squad,  # type: ignore[arg-type]
        tag="tag",
        gameweeks=[3, 4],
        root_gw=3,
        season="2526",
        progress=count_step,
    )
    plan = DEFAULT_STRATEGIES.create(request.move).propose(request)
    assert plan == Proposal(squad, [], [])
    # one step per candidate squad considered, and this strategy considers one
    assert steps == DEFAULT_STRATEGIES.create(request.move).num_increments(request)


@pytest.mark.parametrize(
    ("chip", "bench_boost", "triple_captain"),
    [
        (None, None, None),
        (Chip.BENCH_BOOST, 3, None),
        (Chip.TRIPLE_CAPTAIN, None, 3),
    ],
)
def test_request_resolves_the_chip_gameweeks(chip, bench_boost, triple_captain):
    request = TransferRequest(
        move=GameweekMove(1, chip),
        squad=object(),  # type: ignore[arg-type]
        tag="tag",
        gameweeks=[3, 4, 5],
        root_gw=3,
        season="2526",
    )
    # chips apply to the gameweek being transferred for, not the whole window
    assert request.transfer_gameweek == 3
    assert request.bench_boost_gw == bench_boost
    assert request.triple_captain_gw == triple_captain


@pytest.mark.parametrize(
    ("move", "expected"),
    [
        (GameweekMove(0), 1),
        (GameweekMove(1), 15),
        (GameweekMove(2), 105),
        (GameweekMove(3), 100),
        (GameweekMove(chip=Chip.WILDCARD), 100),
        (GameweekMove(chip=Chip.FREE_HIT), 100),
        (GameweekMove(1, Chip.BENCH_BOOST), 15),
        (GameweekMove(2, Chip.TRIPLE_CAPTAIN), 105),
    ],
)
def test_a_strategy_sizes_its_own_progress_bar(move, expected):
    """
    Every shipped strategy can say how many candidate squads it will consider.

    The number comes from the request, so it cannot drift away from what
    `propose` given the same request actually does.
    """
    request = TransferRequest(
        move=move,
        squad=object(),  # type: ignore[arg-type]
        tag="tag",
        gameweeks=[3, 4],
        root_gw=3,
        season="2526",
        num_iterations=100,
    )
    assert strategy_total(DEFAULT_STRATEGIES.create(move), request) == expected
