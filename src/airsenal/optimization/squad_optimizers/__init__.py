"""
Whole-squad optimizers: one module per way of picking fifteen players.

`SQUAD_OPTIMIZERS` maps a `--squad-optimizer` name to a zero-argument factory.
"""

from collections.abc import Callable
from dataclasses import replace

from airsenal.core.lookup import lookup
from airsenal.optimization.protocols import SquadOptimizer
from airsenal.optimization.squad_optimizers.genetic import (
    GeneticAlgorithmConfig,
    GeneticSquadOptimizer,
)

DEFAULT_SQUAD_OPTIMIZER = "genetic"

SQUAD_OPTIMIZERS: dict[str, Callable[[], SquadOptimizer]] = {
    "genetic": GeneticSquadOptimizer,
}


def build_squad_optimizer(
    name: str = DEFAULT_SQUAD_OPTIMIZER,
    *,
    num_generations: int | None = None,
    population_size: int | None = None,
) -> SquadOptimizer:
    """
    The named whole-squad optimizer, sized by the two flags that describe it.

    `--num-generations` and `--population-size` describe a genetic algorithm, so
    like the tree search's flags they only reach the one component they are about.
    The rest of the GA's defaults live in `GeneticAlgorithmConfig` and nowhere
    else.
    """
    if name != DEFAULT_SQUAD_OPTIMIZER:
        return lookup(SQUAD_OPTIMIZERS, name, "squad optimizer")()
    config = GeneticAlgorithmConfig()
    if num_generations is not None:
        config = replace(config, generations=num_generations)
    if population_size is not None:
        config = replace(config, population_size=population_size)
    return GeneticSquadOptimizer(config)


__all__ = [
    "DEFAULT_SQUAD_OPTIMIZER",
    "SQUAD_OPTIMIZERS",
    "GeneticAlgorithmConfig",
    "GeneticSquadOptimizer",
    "build_squad_optimizer",
]
