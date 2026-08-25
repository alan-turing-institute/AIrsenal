"""
The genetic algorithm, behind the squad-optimizer interface.

A thin wrapper rather than a rewrite: all contact with DEAP stays inside
`genetic_algorithm.py`, which is the one module exempted from mypy's
`disallow_untyped_calls`. If this module ever needs that exemption too, DEAP has
leaked out of the place that is allowed to know about it.
"""

from airsenal.optimization.protocols import SquadRequest
from airsenal.optimization.squad_optimizers.genetic_algorithm import (
    GeneticAlgorithmConfig,
    make_new_squad,
)
from airsenal.squad.squad import Squad

__all__ = ["GeneticAlgorithmConfig", "GeneticSquadOptimizer"]


class GeneticSquadOptimizer:
    """Picks a squad with a DEAP genetic algorithm."""

    def __init__(self, config: GeneticAlgorithmConfig | None = None) -> None:
        self.config = config if config is not None else GeneticAlgorithmConfig()

    def _config_for(self, effort: int | None) -> GeneticAlgorithmConfig:
        """
        This optimizer's settings, scaled to an effort budget if one was given.

        Population and generations both come off the one number, which is
        questionable - they control different things - but it is the only knob the
        wildcard and free-hit path has, and doing it here rather than in a factory
        the caller supplies means the caller does not have to know.
        """
        return self.config if effort is None else self.config.scaled(effort)

    def num_increments(self, effort: int | None = None) -> int:
        # One step per generation: how many individuals a generation evaluates
        # depends on which ones crossover and mutation touched, so generations are
        # the only unit whose count is known before the search starts.
        return self._config_for(effort).generations

    def optimize(self, request: SquadRequest) -> Squad:
        config = self._config_for(request.effort)
        return make_new_squad(
            request.gameweeks,
            tag=request.tag,
            budget=request.scoring.budget,
            season=request.season,
            remove_zero=request.remove_zero,
            sub_weights=request.scoring.sub_weights,
            dummy_sub_cost=request.scoring.dummy_sub_cost,
            bench_boost_gw=request.bench_boost_gw,
            triple_captain_gw=request.triple_captain_gw,
            ga_config=config,
            # Only hand over a reporter when something is watching: given one,
            # SquadOpt.optimize runs the generations itself rather than calling
            # eaSimple once. The result is the same, but an always-live callback
            # would silently change which path an unwatched search takes.
            on_generation=(
                request.advance_progress if request.progress is not None else None
            ),
            dbsession=request.dbsession,
        )
