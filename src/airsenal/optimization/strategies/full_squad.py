"""
Pick a whole new squad, as a wildcard or free hit does.

Every player can change, so this hands off to the genetic algorithm rather than
enumerating swaps.
"""

from airsenal.core.logging import get_logger
from airsenal.optimization.config import GeneticAlgorithmConfig
from airsenal.optimization.moves import GameweekMove
from airsenal.optimization.protocols import TransferPlan, TransferRequest
from airsenal.optimization.squad_ga import make_new_squad
from airsenal.optimization.strategies.registry import TRANSFER_STRATEGIES

logger = get_logger(__name__)


class FullSquadStrategy:
    """Rebuild the squad from scratch within its sale value."""

    def __init__(self, ga_config: GeneticAlgorithmConfig) -> None:
        self.ga_config = ga_config

    def num_increments(self, move: GameweekMove, num_iterations: int) -> int:  # noqa: ARG002
        return num_iterations

    def propose(self, request: TransferRequest) -> TransferPlan:
        move = request.move
        players_out = [p.player_id for p in request.squad.players]
        budget = request.squad.sale_value(request.root_gw, use_api=False)

        gameweeks = request.gameweeks
        if not move.carry_forward:
            # a free hit is reverted afterwards, so only this week's score matters
            gameweeks = [request.transfer_gameweek]

        new_squad = make_new_squad(
            gameweeks,
            tag=request.tag,
            budget=budget,
            season=request.season,
            verbose=False,
            bench_boost_gw=request.bench_boost_gw,
            triple_captain_gw=request.triple_captain_gw,
            ga_config=self.ga_config.scaled(request.num_iterations),
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
    return FullSquadStrategy(config)
