"""
Pick a whole new squad, as a wildcard or free hit does.

Every player can change, so this hands off to a whole-squad optimizer rather than
enumerating swaps. Which one is a property of the instance: the class knows only
the `SquadOptimizer` interface, so swapping the genetic algorithm for something
else here is a change to the registration below, not to the strategy.
"""

from dataclasses import replace

from airsenal.core.logging import get_logger
from airsenal.optimization.config import GeneticAlgorithmConfig, SquadScoringConfig
from airsenal.optimization.moves import GameweekMove
from airsenal.optimization.protocols import (
    SquadOptimizer,
    SquadRequest,
    TransferPlan,
    TransferRequest,
)
from airsenal.optimization.squad_optimizers import GeneticSquadOptimizer
from airsenal.optimization.strategies.registry import TRANSFER_STRATEGIES

logger = get_logger(__name__)


class FullSquadStrategy:
    """Rebuild the squad from scratch within its sale value."""

    def __init__(self, optimizer: SquadOptimizer) -> None:
        self.optimizer = optimizer

    def num_increments(self, move: GameweekMove, num_iterations: int) -> int:  # noqa: ARG002
        return self.optimizer.scaled(num_iterations).num_increments()

    def propose(self, request: TransferRequest) -> TransferPlan:
        move = request.move
        players_out = [p.player_id for p in request.squad.players]
        budget = request.squad.sale_value(request.root_gw, use_api=False)

        gameweeks = request.gameweeks
        if not move.carry_forward:
            # a free hit is reverted afterwards, so only this week's score matters
            gameweeks = [request.transfer_gameweek]

        new_squad = self.optimizer.scaled(request.num_iterations).optimize(
            SquadRequest(
                gameweeks=gameweeks,
                tag=request.tag,
                season=request.season,
                scoring=replace(SquadScoringConfig(), budget=budget),
                bench_boost_gw=request.bench_boost_gw,
                triple_captain_gw=request.triple_captain_gw,
                # the score so far is left out here: a worker's bar is one line among
                # several, labelled by the strategy it is running, and the standalone
                # squad optimisation is the one with a bar to itself to report into
                progress=lambda _best_score: request.advance_progress(),
            )
        )
        players_in = [p.player_id for p in new_squad.players]
        return TransferPlan(
            new_squad,
            # a player kept from the old squad is not a transfer
            players_in=[p for p in players_in if p not in players_out],
            players_out=[p for p in players_out if p not in players_in],
        )


@TRANSFER_STRATEGIES.register("full_squad", GeneticAlgorithmConfig)
def _make(config: GeneticAlgorithmConfig) -> FullSquadStrategy:
    return FullSquadStrategy(GeneticSquadOptimizer(config))
