"""
Transfer optimizers, one per module, each registering itself by name.

Importing this package registers them all, which is what makes
`TRANSFER_OPTIMIZERS.create(name)` work from anywhere.
"""

# importing each module runs its registration
from airsenal.optimization.transfer_optimizers import tree_search
from airsenal.optimization.transfer_optimizers.registry import TRANSFER_OPTIMIZERS
from airsenal.optimization.transfer_optimizers.tree_search import (
    TreeSearchConfig,
    TreeSearchOptimizer,
)

__all__ = [
    "TRANSFER_OPTIMIZERS",
    "TreeSearchConfig",
    "TreeSearchOptimizer",
    "tree_search",
]
