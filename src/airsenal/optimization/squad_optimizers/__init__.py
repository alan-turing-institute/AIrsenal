"""Whole-squad optimizers: one module per way of picking fifteen players."""

from collections.abc import Callable

from airsenal.optimization.protocols import SquadOptimizer
from airsenal.optimization.squad_optimizers.genetic import GeneticSquadOptimizer

DEFAULT_SQUAD_OPTIMIZER = "genetic"

SQUAD_OPTIMIZERS: dict[str, Callable[[], SquadOptimizer]] = {
    "genetic": GeneticSquadOptimizer,
}

__all__ = [
    "DEFAULT_SQUAD_OPTIMIZER",
    "SQUAD_OPTIMIZERS",
    "GeneticSquadOptimizer",
]
