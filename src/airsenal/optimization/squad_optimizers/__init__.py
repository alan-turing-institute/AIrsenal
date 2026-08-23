"""
Whole-squad optimizers, one per module, each registering itself by name.

Importing this package registers them all, which is what makes
`SQUAD_OPTIMIZERS.create(name)` work from anywhere.
"""

# importing each module runs its registration
from airsenal.optimization.squad_optimizers import genetic
from airsenal.optimization.squad_optimizers.genetic import GeneticSquadOptimizer
from airsenal.optimization.squad_optimizers.registry import SQUAD_OPTIMIZERS

__all__ = ["SQUAD_OPTIMIZERS", "GeneticSquadOptimizer", "genetic"]
