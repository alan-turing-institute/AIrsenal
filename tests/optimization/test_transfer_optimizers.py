"""
The seam between "choose transfers for a range of gameweeks" and "do it by
walking the whole strategy tree", and the StrategySet that decides which
strategy handles each move.
"""

import pickle

import pytest

from airsenal.core.enums import Chip
from airsenal.core.registry import ConfigError, lookup
from airsenal.optimization.moves import (
    MAX_FREE_TRANSFERS,
    GameweekMove,
    TransferConstraints,
)
from airsenal.optimization.strategies import (
    DEFAULT_STRATEGIES,
    StrategySet,
)
from airsenal.optimization.strategies.none import NoTransfersStrategy
from airsenal.optimization.strategy import (
    GameweekOutcome,
    Strategy,
    TransferSearchResult,
)
from airsenal.optimization.transfer_optimizers import (
    TRANSFER_OPTIMIZERS,
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
    return Strategy(root_gameweek=1, outcomes=outcomes)


# --------------------------- StrategySet ---------------------------


def test_the_default_set_matches_the_mapping_the_search_has_always_used():
    names = [DEFAULT_STRATEGIES.name_for(GameweekMove(n_transfers=n)) for n in range(5)]
    assert names == ["none", "single", "double", "random", "random"]


def test_a_squad_rebuilding_chip_goes_to_the_whole_squad_strategy():
    assert DEFAULT_STRATEGIES.name_for(GameweekMove(chip=Chip.WILDCARD)) == "full_squad"
    assert DEFAULT_STRATEGIES.name_for(GameweekMove(chip=Chip.FREE_HIT)) == "full_squad"


def test_a_set_can_route_a_move_to_a_different_strategy():
    """The seam: swapping which algorithm handles a move edits no call site."""
    strategies = StrategySet(rebuild="none")
    strategy = strategies.create(GameweekMove(chip=Chip.WILDCARD))

    assert isinstance(strategy, NoTransfersStrategy)
    # and the default set is unaffected
    assert DEFAULT_STRATEGIES.name_for(GameweekMove(chip=Chip.WILDCARD)) == "full_squad"


def test_an_unknown_strategy_name_is_rejected_when_the_set_is_used():
    strategies = StrategySet(rebuild="nope")
    with pytest.raises(ConfigError, match=r"Unknown transfer strategy 'nope'"):
        strategies.create(GameweekMove(chip=Chip.WILDCARD))


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
    # This one used to be dropped on the way to the workers: the Process args
    # tuple was one element shorter than the worker signature, so every worker
    # silently used its own default rather than what the caller asked for.
    assert constraints.max_free_transfers == MAX_FREE_TRANSFERS


def test_constraints_survive_a_pickle():
    constraints = TransferConstraints(max_total_hit=8, max_free_transfers=3)
    assert pickle.loads(pickle.dumps(constraints)) == constraints


# --------------------------- results ---------------------------


def test_the_best_strategy_is_the_highest_scoring_one():
    worse = _strategy(GameweekMove(n_transfers=1), points=10.0)
    better = _strategy(GameweekMove(n_transfers=2), points=20.0)

    result = TransferSearchResult.from_strategies([worse, better])
    assert result.best is better


def test_the_baseline_is_the_strategy_that_does_nothing():
    baseline = _strategy(GameweekMove(), points=5.0)
    active = _strategy(GameweekMove(n_transfers=1), points=20.0)

    result = TransferSearchResult.from_strategies([active, baseline])
    assert result.baseline is baseline
    assert result.baseline_score == 5.0


def test_a_missing_baseline_scores_zero_rather_than_failing():
    """It can legitimately be absent when unused transfers are excluded."""
    result = TransferSearchResult.from_strategies(
        [_strategy(GameweekMove(n_transfers=1), points=20.0)]
    )
    assert result.baseline is None
    assert result.baseline_score == 0.0


def test_every_strategy_considered_is_kept_for_the_dump():
    strategies = [
        _strategy(GameweekMove(), points=5.0),
        _strategy(GameweekMove(n_transfers=1), points=20.0),
    ]
    assert len(TransferSearchResult.from_strategies(strategies).considered) == 2


def test_finding_no_strategy_at_all_is_an_error():
    with pytest.raises(ValueError, match="Failed to find a strategy"):
        TransferSearchResult.from_strategies([])


# --------------------------- the optimizer ---------------------------


def test_the_tree_search_is_registered():
    assert "tree_search" in TRANSFER_OPTIMIZERS


def test_an_unknown_optimizer_names_the_valid_ones():
    with pytest.raises(
        ConfigError, match=r"Unknown transfer optimizer 'nope'.*tree_search"
    ):
        lookup(TRANSFER_OPTIMIZERS, "nope", "transfer optimizer")


def test_the_tree_search_defaults_its_own_config():
    """Every table entry has to be constructible with no arguments."""
    assert TRANSFER_OPTIMIZERS["tree_search"]().config == TreeSearchConfig()


def test_the_config_reaches_the_optimizer():
    optimizer = TreeSearchOptimizer(TreeSearchConfig(num_thread=2))
    assert optimizer.config.num_thread == 2


def test_the_config_defaults_to_the_default_strategy_set():
    assert TreeSearchConfig().strategies == DEFAULT_STRATEGIES
