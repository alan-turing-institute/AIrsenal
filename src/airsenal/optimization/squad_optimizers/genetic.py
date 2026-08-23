"""
The genetic algorithm, behind the squad-optimizer interface.

A thin wrapper rather than a rewrite: all contact with DEAP stays inside
`optimization/squad_ga.py`, which is the one module exempted from mypy's
`disallow_untyped_calls`. If this module ever needs that exemption too, DEAP has
leaked out of the place that is allowed to know about it.
"""

from airsenal.optimization.config import GeneticAlgorithmConfig
from airsenal.optimization.protocols import SquadRequest
from airsenal.optimization.squad_ga import make_new_squad
from airsenal.optimization.squad_optimizers.registry import SQUAD_OPTIMIZERS
from airsenal.squad.squad import Squad


class GeneticSquadOptimizer:
    """Picks a squad with a DEAP genetic algorithm."""

    def __init__(self, config: GeneticAlgorithmConfig) -> None:
        self.config = config

    def num_increments(self) -> int:
        # One step per generation: how many individuals a generation evaluates
        # depends on which ones crossover and mutation touched, so generations are
        # the only unit whose count is known before the search starts.
        return self.config.generations

    def scaled(self, num_iterations: int) -> "GeneticSquadOptimizer":
        return GeneticSquadOptimizer(self.config.scaled(num_iterations))

    def optimize(self, request: SquadRequest) -> Squad:
        return make_new_squad(
            request.gameweeks,
            tag=request.tag,
            budget=request.scoring.budget,
            season=request.season,
            remove_zero=request.remove_zero,
            sub_weights=request.scoring.sub_weights.as_dict(),
            dummy_sub_cost=request.scoring.dummy_sub_cost,
            bench_boost_gw=request.bench_boost_gw,
            triple_captain_gw=request.triple_captain_gw,
            ga_config=self.config,
            # make_new_squad still defaults this to True and overrides the config
            # with it, so leaving it out would turn DEAP's per-generation logbook
            # on underneath a progress bar.
            verbose=self.config.verbose,
            # Only hand over a reporter when something is watching: given one,
            # SquadOpt.optimize runs the generations itself rather than calling
            # eaSimple once. The result is the same, but an always-live callback
            # would silently change which path an unwatched search takes.
            on_generation=(
                request.advance_progress if request.progress is not None else None
            ),
            dbsession=request.dbsession,
        )


@SQUAD_OPTIMIZERS.register("genetic", GeneticAlgorithmConfig)
def _make(config: GeneticAlgorithmConfig) -> GeneticSquadOptimizer:
    return GeneticSquadOptimizer(config)
