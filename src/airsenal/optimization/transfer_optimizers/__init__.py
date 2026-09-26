"""
Transfer optimizers: one module per way of searching a whole gameweek window.

`TRANSFER_OPTIMIZERS` maps a `--transfer-optimizer` name to a factory. Like the
team models' table, its factories are not zero-argument: each takes the search
flags as optional keyword-only arguments, applies the ones it has, and rejects a
flag it has no use for rather than ignoring it.
"""

from collections.abc import Callable
from dataclasses import replace

from airsenal.core.lookup import ConfigError, lookup
from airsenal.optimization.protocols import TransferOptimizer
from airsenal.optimization.transfer_optimizers.auto import AutoOptimizer
from airsenal.optimization.transfer_optimizers.mcts import MCTSConfig, MCTSOptimizer
from airsenal.optimization.transfer_optimizers.tree_search import (
    TreeSearchConfig,
    TreeSearchOptimizer,
)

DEFAULT_TRANSFER_OPTIMIZER = "auto"


def _tree_search(
    *,
    num_thread: int | None = None,
    num_iterations: int | None = None,
    max_expansions: int | None = None,
    profile: bool = False,
) -> TreeSearchOptimizer:
    if max_expansions is not None:
        msg = "--max-expansions is the MCTS budget: the tree search makes every node"
        raise ConfigError(msg)
    config = TreeSearchConfig(profile=profile)
    if num_thread is not None:
        config = replace(config, num_thread=num_thread)
    if num_iterations is not None:
        config = replace(config, num_iterations=num_iterations)
    return TreeSearchOptimizer(config)


def _mcts(
    *,
    num_thread: int | None = None,
    num_iterations: int | None = None,
    max_expansions: int | None = None,
    profile: bool = False,
) -> MCTSOptimizer:
    if profile:
        msg = "--profile only profiles the tree search's workers"
        raise ConfigError(msg)
    config = MCTSConfig()
    if num_thread is not None:
        config = replace(config, num_thread=num_thread)
    if num_iterations is not None:
        config = replace(config, num_iterations=num_iterations)
    if max_expansions is not None:
        config = replace(config, max_expansions=max_expansions)
    return MCTSOptimizer(config)


def _auto(
    *,
    num_thread: int | None = None,
    num_iterations: int | None = None,
    max_expansions: int | None = None,
    profile: bool = False,
) -> AutoOptimizer:
    """Each flag reaches whichever of the two searches has it."""
    return AutoOptimizer(
        tree_search=_tree_search(
            num_thread=num_thread, num_iterations=num_iterations, profile=profile
        ),
        mcts=_mcts(
            num_thread=num_thread,
            num_iterations=num_iterations,
            max_expansions=max_expansions,
        ),
    )


TRANSFER_OPTIMIZERS: dict[str, Callable[..., TransferOptimizer]] = {
    "auto": _auto,
    "mcts": _mcts,
    "tree_search": _tree_search,
}


def build_transfer_optimizer(
    name: str = DEFAULT_TRANSFER_OPTIMIZER,
    *,
    num_thread: int | None = None,
    num_iterations: int | None = None,
    max_expansions: int | None = None,
    profile: bool = False,
) -> TransferOptimizer:
    """The named transfer search, configured from the CLI flags that describe it."""
    return lookup(TRANSFER_OPTIMIZERS, name, "transfer optimizer")(
        num_thread=num_thread,
        num_iterations=num_iterations,
        max_expansions=max_expansions,
        profile=profile,
    )


__all__ = [
    "DEFAULT_TRANSFER_OPTIMIZER",
    "TRANSFER_OPTIMIZERS",
    "AutoOptimizer",
    "MCTSConfig",
    "MCTSOptimizer",
    "TreeSearchConfig",
    "TreeSearchOptimizer",
    "build_transfer_optimizer",
]
