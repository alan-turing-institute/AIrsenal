"""Replace two players, trying every pair in turn."""

import copy
from functools import partial
from typing import TYPE_CHECKING

from airsenal.core.logging import get_logger
from airsenal.db.queries.predictions import get_predicted_points
from airsenal.game.enums import Position
from airsenal.game.season import CURRENT_SEASON
from airsenal.optimization.protocols import (
    Proposal,
    StepCounter,
    TransferRequest,
)
from airsenal.optimization.squad_score import get_discounted_squad_score
from airsenal.squad.player import CandidateCache
from airsenal.squad.squad import Squad, SubWeights

if TYPE_CHECKING:
    from airsenal.db.models import Player

logger = get_logger(__name__)

# 15 players choose 2, ignoring order: 15*14/2 = 105 candidate squads.
NUM_PAIRS = 105


def make_optimum_double_transfer(
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
    candidates: CandidateCache | None = None,
) -> tuple[Squad, list[int], list[int]]:
    """
    Try every pair of transfers in turn, which is affordable for just two.

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
    ordered_player_lists: dict[str, list[tuple[Player, float]]] = {
        pos: get_predicted_points(
            gameweeks=gameweeks, position=pos, tag=tag, season=season
        )
        for pos in list(Position.back_to_front())
    }
    candidates = CandidateCache.reuse(candidates, root_gameweek, squad.season)
    for i in range(len(squad.players) - 1):
        pout_1 = squad.players[i]

        new_squad_remove_1 = squad.copy()
        new_squad_remove_1.remove_player(pout_1.player_id, gameweek=root_gameweek)
        for j in range(i + 1, len(squad.players)):
            if on_step:
                on_step()

            pout_2 = squad.players[j]
            new_squad_remove_2 = new_squad_remove_1.copy()
            new_squad_remove_2.remove_player(pout_2.player_id, gameweek=root_gameweek)
            logger.debug("Removing players %s %s", i, j)
            # what positions do we need to fill?
            positions_needed = [pout_1.position, pout_2.position]

            # now loop over lists of players and add players back in, checking
            # each fits before copying a squad to add them to
            for pin_1, _ in ordered_player_lists[positions_needed[0]]:
                if pin_1.player_id in [pout_1.player_id, pout_2.player_id]:
                    continue  # no point in adding same player back in
                candidate_1 = candidates.get(pin_1)
                if not new_squad_remove_2.can_add(candidate_1):
                    continue
                new_squad_add_1 = new_squad_remove_2.copy()
                new_squad_add_1.add_player(
                    copy.copy(candidate_1), gameweek=root_gameweek
                )
                for pin_2, _ in ordered_player_lists[positions_needed[1]]:
                    if pin_2.player_id in [
                        pin_1.player_id,
                        pout_1.player_id,
                        pout_2.player_id,
                    ]:
                        continue  # no point in adding same player back in
                    candidate_2 = candidates.get(pin_2)
                    if not new_squad_add_1.can_add(candidate_2):
                        continue
                    new_squad_add_2 = new_squad_add_1.copy()
                    new_squad_add_2.add_player(
                        copy.copy(candidate_2), gameweek=root_gameweek
                    )
                    total_points = score(new_squad_add_2)
                    if total_points > best_score:
                        best_score = total_points
                        best_pid_out = [pout_1.player_id, pout_2.player_id]
                        best_pid_in = [pin_1.player_id, pin_2.player_id]
                        best_squad = new_squad_add_2
                    break

    if best_squad is None:
        msg = "Failed to find valid double transfer for squad"
        raise RuntimeError(msg)

    return best_squad, best_pid_out, best_pid_in


class DoubleTransferStrategy:
    """Try replacing each pair of players in turn and keep the best."""

    def num_increments(self, request: TransferRequest) -> int:  # noqa: ARG002
        return NUM_PAIRS

    def propose(self, request: TransferRequest) -> Proposal:
        squad, players_out, players_in = make_optimum_double_transfer(
            request.squad,
            request.tag,
            request.gameweeks,
            request.root_gameweek,
            request.season,
            on_step=request.progress,
            bench_boost_gameweek=request.bench_boost_gameweek,
            triple_captain_gameweek=request.triple_captain_gameweek,
            sub_weights=request.scoring.sub_weights,
            candidates=request.candidates,
        )
        return Proposal(squad, players_in, players_out)
