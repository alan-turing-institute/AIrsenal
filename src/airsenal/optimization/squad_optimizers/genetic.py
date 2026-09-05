"""The genetic algorithm, behind the squad-optimizer interface."""

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
        """This optimizer's settings, scaled to an effort budget if one was given."""
        return self.config if effort is None else self.config.scaled(effort)

    def num_increments(self, effort: int | None = None) -> int:
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
            on_generation=(
                request.advance_progress if request.progress is not None else None
            ),
            dbsession=request.dbsession,
        )
