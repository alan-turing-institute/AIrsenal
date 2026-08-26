"""Replace two players, trying every pair in turn."""

from typing import TYPE_CHECKING

from airsenal.core.copy import fastcopy
from airsenal.core.logging import get_logger
from airsenal.db.queries.gameweeks import next_gameweek
from airsenal.db.queries.predictions import get_predicted_points
from airsenal.game.enums import Position
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

# 15 players choose 2, ignoring order: 15*14/2 = 105 candidate squads.
NUM_PAIRS = 105


def make_optimum_double_transfer(
    squad: Squad,
    tag: str,
    gameweeks: list[int] | None = None,
    root_gw: int | None = None,
    season: str = CURRENT_SEASON,
    on_step: StepCounter | None = None,
    bench_boost_gw: int | None = None,
    triple_captain_gw: int | None = None,
    sub_weights: SubWeights | None = None,
) -> tuple[Squad, list[int], list[int]]:
    """
    Try every pair of transfers in turn, which is affordable for just two.

    Candidates are ordered by their total expected points over `gameweeks`.
    """
    if not gameweeks:
        gameweeks = [next_gameweek()]
        root_gw = next_gameweek()

    transfer_gw = min(gameweeks)  # the week we're making the transfer
    best_score = -1.0
    best_squad = None
    best_pid_out, best_pid_in = [], []
    ordered_player_lists: dict[str, list[tuple[Player, float]]] = {
        pos: get_predicted_points(
            gameweeks=gameweeks, position=pos, tag=tag, season=season
        )
        for pos in list(Position.back_to_front())
    }
    for i in range(len(squad.players) - 1):
        positions_needed = []
        pout_1 = squad.players[i]

        new_squad_remove_1 = fastcopy(squad)
        new_squad_remove_1.remove_player(pout_1.player_id, gameweek=transfer_gw)
        for j in range(i + 1, len(squad.players)):
            if on_step:
                on_step()

            pout_2 = squad.players[j]
            new_squad_remove_2 = fastcopy(new_squad_remove_1)
            new_squad_remove_2.remove_player(pout_2.player_id, gameweek=transfer_gw)
            logger.debug("Removing players %s %s", i, j)
            # what positions do we need to fill?
            positions_needed = [pout_1.position, pout_2.position]

            # now loop over lists of players and add players back in
            for pin_1 in ordered_player_lists[positions_needed[0]]:
                if pin_1[0].player_id in [pout_1.player_id, pout_2.player_id]:
                    continue  # no point in adding same player back in
                new_squad_add_1 = fastcopy(new_squad_remove_2)
                added_1_ok = new_squad_add_1.add_player(pin_1[0], gameweek=transfer_gw)
                if not added_1_ok:
                    continue
                for pin_2 in ordered_player_lists[positions_needed[1]]:
                    new_squad_add_2 = fastcopy(new_squad_add_1)
                    if (
                        pin_2[0] == pin_1[0]
                        or pin_2[0].player_id == pout_1.player_id
                        or pin_2[0].player_id == pout_2.player_id
                    ):
                        continue  # no point in adding same player back in
                    added_2_ok = new_squad_add_2.add_player(
                        pin_2[0], gameweek=transfer_gw
                    )
                    if added_2_ok:
                        # calculate the score
                        total_points = get_discounted_squad_score(
                            new_squad_add_2,
                            gameweeks,
                            tag,
                            root_gw=root_gw,
                            bench_boost_gw=bench_boost_gw,
                            triple_captain_gw=triple_captain_gw,
                            sub_weights=sub_weights,
                        )
                        if total_points > best_score:
                            best_score = total_points
                            best_pid_out = [pout_1.player_id, pout_2.player_id]
                            best_pid_in = [pin_1[0].player_id, pin_2[0].player_id]
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
            request.root_gw,
            request.season,
            on_step=request.progress,
            bench_boost_gw=request.bench_boost_gw,
            triple_captain_gw=request.triple_captain_gw,
            sub_weights=request.scoring.sub_weights,
        )
        return Proposal(squad, players_in, players_out)
