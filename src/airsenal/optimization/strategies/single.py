"""Replace one player, trying every possibility in turn."""

from functools import partial
from typing import TYPE_CHECKING

from airsenal.core.copy import fastcopy
from airsenal.core.logging import get_logger
from airsenal.db.queries.predictions import get_predicted_points
from airsenal.game.enums import Position
from airsenal.game.scoring import SQUAD_SIZE
from airsenal.game.season import CURRENT_SEASON
from airsenal.optimization.protocols import (
    Proposal,
    StepCounter,
    TransferRequest,
)
from airsenal.optimization.squad_score import get_discounted_squad_score
from airsenal.squad.squad import Squad, SubWeights

if TYPE_CHECKING:
    from airsenal.db.models import Player

logger = get_logger(__name__)


def make_optimum_single_transfer(
    squad: Squad,
    tag: str,
    gameweeks: list[int],
    root_gameweek: int | None = None,
    season: str = CURRENT_SEASON,
    on_step: StepCounter | None = None,
    bench_boost_gameweek: int | None = None,
    triple_captain_gameweek: int | None = None,
    *,
    sub_weights: SubWeights,
) -> tuple[Squad, list[int], list[int]]:
    """
    Try every single transfer in turn, which is affordable for just one.

    Candidates are ordered by their total expected points over `gameweeks`, and
    priced as at `root_gameweek`, which is what `TransferRequest.root_gameweek`
    documents.
    """
    if root_gameweek is None:
        root_gameweek = min(gameweeks)

    score = partial(
        get_discounted_squad_score,
        gameweeks=gameweeks,
        tag=tag,
        root_gameweek=root_gameweek,
        bench_boost_gameweek=bench_boost_gameweek,
        triple_captain_gameweek=triple_captain_gameweek,
        sub_weights=sub_weights,
    )
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
        new_squad.remove_player(p_out.player_id, gameweek=root_gameweek)
        for p_in in ordered_player_lists[position]:
            if p_in[0].player_id == p_out.player_id:
                continue  # no point in adding the same player back in
            added_ok = new_squad.add_player(p_in[0], gameweek=root_gameweek)
            if added_ok:
                logger.debug("Added player %s", p_in[0])
                total_points = score(new_squad)
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

    def num_increments(self, request: TransferRequest) -> int:  # noqa: ARG002
        return SQUAD_SIZE

    def propose(self, request: TransferRequest) -> Proposal:
        squad, players_out, players_in = make_optimum_single_transfer(
            request.squad,
            request.tag,
            request.gameweeks,
            request.root_gameweek,
            request.season,
            on_step=request.progress,
            bench_boost_gameweek=request.bench_boost_gameweek,
            triple_captain_gameweek=request.triple_captain_gameweek,
            sub_weights=request.scoring.sub_weights,
        )
        return Proposal(squad, players_in, players_out)
