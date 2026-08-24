"""Transfer optimizers: one module per way of searching a whole gameweek window."""

from collections.abc import Callable

from airsenal.optimization.protocols import TransferOptimizer
from airsenal.optimization.transfer_optimizers.tree_search import (
    TreeSearchConfig,
    TreeSearchOptimizer,
)

TRANSFER_OPTIMIZERS: dict[str, Callable[[], TransferOptimizer]] = {
    "tree_search": TreeSearchOptimizer,
}

__all__ = [
    "TRANSFER_OPTIMIZERS",
    "TreeSearchConfig",
    "TreeSearchOptimizer",
]
