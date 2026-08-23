"""Replace one player, trying every possibility in turn."""

from typing import TYPE_CHECKING

from airsenal.core.copy import fastcopy
from airsenal.core.enums import Position
from airsenal.core.logging import get_logger
from airsenal.core.season import CURRENT_SEASON
from airsenal.db.queries.gameweeks import next_gameweek
from airsenal.db.queries.predictions import get_predicted_points
from airsenal.optimization.moves import GameweekMove
from airsenal.optimization.protocols import (
    Proposal,
    StepCounter,
    TransferRequest,
)
from airsenal.optimization.squad_score import get_discounted_squad_score
from airsenal.squad.squad import Squad

if TYPE_CHECKING:
    from airsenal.db.models import Player

logger = get_logger(__name__)

# 15 players in a squad, so 15 candidate squads to score.
SQUAD_SIZE = 15


def make_optimum_single_transfer(
    squad: Squad,
    tag: str,
    gameweeks: list[int] | None = None,
    root_gw: int | None = None,
    season: str = CURRENT_SEASON,
    on_step: StepCounter | None = None,
    bench_boost_gw: int | None = None,
    triple_captain_gw: int | None = None,
) -> tuple[Squad, list[int], list[int]]:
    """
    If we want to just make one transfer, it's not unfeasible to try all
    possibilities in turn.

    We will order the list of potential transfers via the sum of
    expected points over a specified range of gameweeks.
    """
    if not gameweeks:
        gameweeks = [next_gameweek()]
        root_gw = next_gameweek()

    transfer_gw = min(gameweeks)  # the week we're making the transfer

    best_score = -1.0
    best_squad = None
    best_pid_out, best_pid_in = [], []

    logger.debug("Creating ordered player lists")
    ordered_player_lists: dict[str, list[tuple[Player, float]]] = {
        pos: get_predicted_points(
            gameweeks=gameweeks, position=pos, tag=tag, season=season
        )
        for pos in list(Position.back_to_front())
    }
    for p_out in squad.players:
        if on_step:
            on_step()

        new_squad = fastcopy(squad)
        position = p_out.position
        logger.debug("Removing player %s", p_out)
        new_squad.remove_player(p_out.player_id, gameweek=transfer_gw)
        for p_in in ordered_player_lists[position]:
            if p_in[0].player_id == p_out.player_id:
                continue  # no point in adding the same player back in
            added_ok = new_squad.add_player(p_in[0], gameweek=transfer_gw)
            if added_ok:
                logger.debug("Added player %s", p_in[0])
                total_points = get_discounted_squad_score(
                    new_squad,
                    gameweeks,
                    tag,
                    root_gw=root_gw,
                    bench_boost_gw=bench_boost_gw,
                    triple_captain_gw=triple_captain_gw,
                )
                if total_points > best_score:
                    best_score = total_points
                    best_pid_out = [p_out.player_id]
                    best_pid_in = [p_in[0].player_id]
                    best_squad = new_squad
                break
            logger.debug("Failed to add %s", p_in[0])
        if not new_squad.is_complete():
            logger.debug("Failed to find a valid replacement for %s", p_out.player_id)

    if best_squad is None:
        msg = "Failed to find valid single transfer for squad"
        raise RuntimeError(msg)

    return best_squad, best_pid_out, best_pid_in


class SingleTransferStrategy:
    """Try replacing each of the 15 players in turn and keep the best."""

    def num_increments(self, move: GameweekMove, num_iterations: int) -> int:  # noqa: ARG002
        return SQUAD_SIZE

    def propose(self, request: TransferRequest) -> Proposal:
        squad, players_out, players_in = make_optimum_single_transfer(
            request.squad,
            request.tag,
            request.gameweeks,
            request.root_gw,
            request.season,
            on_step=request.progress,
            bench_boost_gw=request.bench_boost_gw,
            triple_captain_gw=request.triple_captain_gw,
        )
        return Proposal(squad, players_in, players_out)
