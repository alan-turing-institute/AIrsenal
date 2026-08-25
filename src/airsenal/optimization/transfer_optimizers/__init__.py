"""
Transfer optimizers: one module per way of searching a whole gameweek window.

`TRANSFER_OPTIMIZERS` maps a `--transfer-optimizer` name to a zero-argument
factory.
"""

from collections.abc import Callable
from dataclasses import replace

from airsenal.core.lookup import lookup
from airsenal.optimization.protocols import TransferOptimizer
from airsenal.optimization.transfer_optimizers.tree_search import (
    TreeSearchConfig,
    TreeSearchOptimizer,
)

DEFAULT_TRANSFER_OPTIMIZER = "tree_search"

TRANSFER_OPTIMIZERS: dict[str, Callable[[], TransferOptimizer]] = {
    "tree_search": TreeSearchOptimizer,
}


def build_transfer_optimizer(
    name: str = DEFAULT_TRANSFER_OPTIMIZER,
    *,
    num_thread: int | None = None,
    num_iterations: int | None = None,
    profile: bool = False,
) -> TransferOptimizer:
    """
    The named transfer search, configured from the CLI flags that describe it.

    `--num-thread`, `--num-iterations` and `--profile` are the tree search's own
    settings, so they only reach the tree search; any other optimizer named here
    starts from its own defaults. Finer configuration means constructing the
    component in Python, which is what the protocols are for.
    """
    if name != DEFAULT_TRANSFER_OPTIMIZER:
        return lookup(TRANSFER_OPTIMIZERS, name, "transfer optimizer")()
    config = TreeSearchConfig(profile=profile)
    if num_thread is not None:
        config = replace(config, num_thread=num_thread)
    if num_iterations is not None:
        config = replace(config, num_iterations=num_iterations)
    return TreeSearchOptimizer(config)


__all__ = [
    "DEFAULT_TRANSFER_OPTIMIZER",
    "TRANSFER_OPTIMIZERS",
    "TreeSearchConfig",
    "TreeSearchOptimizer",
    "build_transfer_optimizer",
]
