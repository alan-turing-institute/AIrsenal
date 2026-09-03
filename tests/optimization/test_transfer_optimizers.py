"""
The seam between choosing transfers for a window and walking the strategy tree.

Also the StrategySet that decides which strategy handles each move.
"""

import pickle

import pytest

from airsenal.game.enums import Chip
from airsenal.optimization.moves import MAX_FREE_TRANSFERS, GameweekMove
from airsenal.optimization.plan import (
    GameweekOutcome,
    Plan,
    TransferSearchResult,
)
from airsenal.optimization.protocols import TransferConstraints
from airsenal.optimization.strategies import (
    DEFAULT_STRATEGIES,
    StrategySet,
)
from airsenal.optimization.strategies.none import NoTransfersStrategy
from airsenal.optimization.transfer_optimizers import (
    TreeSearchConfig,
    TreeSearchOptimizer,
)


def _strategy(*moves, points):
    """A finished strategy over the given moves, scoring `points` in the first."""
    outcomes = tuple(
        GameweekOutcome(
            gameweek=1 + i,
            move=move,
            points=points if i == 0 else 0.0,
            discount_factor=1.0,
            points_hit=0,
            free_transfers=1,
        )
        for i, move in enumerate(moves)
    )
    return Plan(root_gameweek=1, outcomes=outcomes)


# --------------------------- StrategySet ---------------------------


def test_a_set_can_route_a_move_to_a_different_strategy():
    """The seam: swapping which algorithm handles a move edits no call site."""
    strategies = StrategySet(rebuild="none")
    strategy = strategies.create(GameweekMove(chip=Chip.WILDCARD))

    assert isinstance(strategy, NoTransfersStrategy)
    # and the default set is unaffected
    assert DEFAULT_STRATEGIES.name_for(GameweekMove(chip=Chip.WILDCARD)) == "full_squad"


def test_a_strategy_set_survives_a_pickle():
    """
    It is handed to forked workers, and Process pickles its arguments off posix.

    Holding names rather than built strategies is what makes that safe - a
    strategy holds an optimizer, which need not be picklable.
    """
    strategies = StrategySet(rebuild="none")
    restored = pickle.loads(pickle.dumps(strategies))

    assert isinstance(
        restored.create(GameweekMove(chip=Chip.WILDCARD)), NoTransfersStrategy
    )


# --------------------------- constraints ---------------------------


def test_constraints_default_to_todays_behaviour():
    constraints = TransferConstraints()

    assert constraints.max_total_hit is None
    assert constraints.allow_unused_transfers is False
    assert constraints.max_opt_transfers == 2
    # Reaches the workers inside this frozen object, so a worker cannot fall
    # back to its own default without the caller noticing.
    assert constraints.max_free_transfers == MAX_FREE_TRANSFERS


def test_constraints_survive_a_pickle():
    constraints = TransferConstraints(max_total_hit=8, max_free_transfers=3)
    assert pickle.loads(pickle.dumps(constraints)) == constraints


# --------------------------- results ---------------------------


def test_the_best_plan_is_the_highest_scoring_one():
    worse = _strategy(GameweekMove(n_transfers=1), points=10.0)
    better = _strategy(GameweekMove(n_transfers=2), points=20.0)

    result = TransferSearchResult.from_plans([worse, better])
    assert result.best is better


def test_the_baseline_is_the_strategy_that_does_nothing():
    baseline = _strategy(GameweekMove(), points=5.0)
    active = _strategy(GameweekMove(n_transfers=1), points=20.0)

    result = TransferSearchResult.from_plans([active, baseline])
    assert result.baseline is baseline
    assert result.baseline_score == 5.0


def test_a_missing_baseline_scores_zero_rather_than_failing():
    """It can legitimately be absent when unused transfers are excluded."""
    result = TransferSearchResult.from_plans(
        [_strategy(GameweekMove(n_transfers=1), points=20.0)]
    )
    assert result.baseline is None
    assert result.baseline_score == 0.0


def test_every_strategy_considered_is_kept_for_the_dump():
    strategies = [
        _strategy(GameweekMove(), points=5.0),
        _strategy(GameweekMove(n_transfers=1), points=20.0),
    ]
    assert len(TransferSearchResult.from_plans(strategies).considered) == 2


def test_the_baseline_on_its_own_is_a_valid_result():
    """
    Doing nothing is an answer, not a failed search.

    Constraints that admit no move - --max-transfers 0 with a full bank of free
    transfers and unused transfers disallowed - leave the tree empty, and the
    baseline the search computes separately is then the only plan there is.
    """
    baseline = _strategy(GameweekMove(), points=5.0)

    result = TransferSearchResult.from_plans([baseline])
    assert result.best is baseline
    assert result.baseline is baseline
    assert result.baseline_score == 5.0


def test_finding_no_plan_at_all_is_an_error():
    with pytest.raises(ValueError, match="Failed to find a plan"):
        TransferSearchResult.from_plans([])


# --------------------------- the optimizer ---------------------------


def test_the_config_reaches_the_optimizer():
    optimizer = TreeSearchOptimizer(TreeSearchConfig(num_thread=2))
    assert optimizer.config.num_thread == 2


def test_the_config_defaults_to_the_default_strategy_set():
    assert TreeSearchConfig().strategies == DEFAULT_STRATEGIES
