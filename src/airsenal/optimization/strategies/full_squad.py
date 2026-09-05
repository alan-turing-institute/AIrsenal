"""
Pick a whole new squad, as a wildcard or free hit does.

Every player can change, so this hands off to a whole-squad optimizer rather
than enumerating swaps.
"""

from dataclasses import replace

from airsenal.core.logging import get_logger
from airsenal.optimization.protocols import (
    Proposal,
    SquadOptimizer,
    SquadRequest,
    TransferRequest,
    progress_total,
)
from airsenal.optimization.squad_optimizers import GeneticSquadOptimizer

logger = get_logger(__name__)


class FullSquadStrategy:
    """Rebuild the squad from scratch within its sale value."""

    @staticmethod
    def _optimizer(request: TransferRequest) -> SquadOptimizer:
        """The whole-squad optimizer this request wants, or the default."""
        return request.squad_optimizer or GeneticSquadOptimizer()

    def num_increments(self, request: TransferRequest) -> int:
        return (
            progress_total(self._optimizer(request), effort=request.num_iterations) or 1
        )

    def propose(self, request: TransferRequest) -> Proposal:
        move = request.move
        players_out = [p.player_id for p in request.squad.players]
        budget = request.squad.sale_value(request.root_gw, use_api=False)

        gameweeks = request.gameweeks
        if not move.carry_forward:
            # a free hit is reverted afterwards, so only this week's score matters
            gameweeks = [request.transfer_gameweek]

        new_squad = self._optimizer(request).optimize(
            SquadRequest(
                gameweeks=gameweeks,
                tag=request.tag,
                season=request.season,
                scoring=replace(request.scoring, budget=budget),
                bench_boost_gw=request.bench_boost_gw,
                triple_captain_gw=request.triple_captain_gw,
                effort=request.num_iterations,
                progress=lambda _best_score: request.advance_progress(),
            )
        )
        players_in = [p.player_id for p in new_squad.players]
        return Proposal(
            new_squad,
            # a player kept from the old squad is not a transfer
            players_in=[p for p in players_in if p not in players_out],
            players_out=[p for p in players_out if p not in players_in],
        )
