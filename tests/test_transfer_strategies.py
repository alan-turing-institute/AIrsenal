"""
Tests for the transfer strategy registry and the move-to-strategy mapping.

These check the wiring, not the searches themselves - the searches are covered
by test_optimization.py.
"""

import pytest

from airsenal.core.enums import Chip
from airsenal.optimization.moves import GameweekMove
from airsenal.optimization.protocols import (
    TransferPlan,
    TransferRequest,
    TransferStrategy,
)
from airsenal.optimization.strategies import (
    TRANSFER_STRATEGIES,
    select_strategy,
    strategy_name_for,
)


def test_all_five_strategies_are_registered():
    assert TRANSFER_STRATEGIES.names() == (
        "double",
        "full_squad",
        "none",
        "random",
        "single",
    )


@pytest.mark.parametrize("name", TRANSFER_STRATEGIES.names())
def test_registered_strategies_satisfy_the_protocol(name):
    strategy = TRANSFER_STRATEGIES.create(name)
    # Protocols are not runtime_checkable here on purpose - isinstance against a
    # Protocol only checks that the names exist, which is the stringly-typed
    # dispatch we are getting rid of. Check the callables directly instead.
    assert callable(strategy.num_increments)
    assert callable(strategy.propose)
    # and that it is usable where a TransferStrategy is expected
    accepts: TransferStrategy = strategy
    assert accepts is strategy


def test_unknown_strategy_lists_the_valid_ones():
    with pytest.raises(ValueError, match="Choose from: double, full_squad"):
        TRANSFER_STRATEGIES.create("wildcard-ish")


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
    assert strategy_name_for(move) == expected
    assert select_strategy(move) is not None


@pytest.mark.parametrize("n_transfers", range(16))
@pytest.mark.parametrize(
    "chip", [None, Chip.BENCH_BOOST, Chip.TRIPLE_CAPTAIN, Chip.WILDCARD, Chip.FREE_HIT]
)
def test_every_move_maps_to_a_registered_strategy(n_transfers, chip):
    if chip is not None and chip.rebuilds_squad:
        move = GameweekMove(chip=chip)
    else:
        move = GameweekMove(n_transfers, chip)
    assert strategy_name_for(move) in TRANSFER_STRATEGIES.names()


def test_none_strategy_keeps_the_squad_and_advances_the_progress_bar():
    calls = []
    squad = object()
    request = TransferRequest(
        move=GameweekMove(0),
        squad=squad,  # type: ignore[arg-type]
        tag="tag",
        gameweeks=[3, 4],
        root_gw=3,
        season="2526",
        progress=(lambda inc, pid: calls.append((inc, pid)), 100.0, 0),  # type: ignore[arg-type]
    )
    plan = select_strategy(request.move).propose(request)
    assert plan == TransferPlan(squad, [], [])
    assert calls == [(100.0, 0)]


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
