"""
Replace three or more players by sampling.

There are too many combinations to enumerate once more than two players change,
so this samples random swaps and keeps the best squad it finds.
"""

import random
from operator import itemgetter
from typing import TYPE_CHECKING

from airsenal.core.copy import fastcopy
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


def make_random_transfers(
    squad: Squad,
    tag: str,
    nsubs: int = 1,
    gameweeks: list[int] | None = None,
    root_gw: int | None = None,
    num_iter: int = 1,
    on_step: StepCounter | None = None,
    season: str = CURRENT_SEASON,
    bench_boost_gw: int | None = None,
    triple_captain_gw: int | None = None,
) -> tuple[Squad, list[int], list[int]]:
    """
    choose nsubs random players to sub out, and then select players
    using a triangular PDF to preferentially select the replacements with
    the best expected score to fill their place.
    Do this num_iter times and choose the best total score over gameweeks gameweeks.
    """
    best_score = -1.0
    best_squad = None
    best_pid_out, best_pid_in = [], []
    max_tries = 100
    for _ in range(num_iter):
        if on_step:
            on_step()

        new_squad = fastcopy(squad)

        if not gameweeks:
            gameweeks = [next_gameweek()]
            root_gw = next_gameweek()

        transfer_gw = min(gameweeks)  # the week we're making the transfer
        players_to_remove: list[int] = []  # this is the index within the squad
        removed_players: list[int] = []  # this is the player_ids
        # order the players in the squad by predicted_points - least-to-most
        player_list: list[tuple[int, float]] = []
        for p in squad.players:
            p.calc_predicted_points(tag)
            player_list.append((p.player_id, p.predicted_points[tag][gameweeks[0]]))
        player_list.sort(key=itemgetter(1), reverse=False)
        while len(players_to_remove) < nsubs:
            index = int(random.triangular(0, len(player_list), 0))
            if index not in players_to_remove:
                players_to_remove.append(index)

        positions_needed = []
        for squad_index in players_to_remove:
            positions_needed.append(squad.players[squad_index].position)
            removed_players.append(squad.players[squad_index].player_id)
            new_squad.remove_player(removed_players[-1], gameweek=transfer_gw)
        predicted_points = {
            pos: get_predicted_points(
                position=pos, gameweeks=gameweeks, tag=tag, season=season
            )
            for pos in set(positions_needed)
        }
        complete_squad = False
        added_players: list[Player] = []
        attempt = 0
        while not complete_squad:
            # sample with a triangular PDF - preferentially select players near
            # the start
            added_players = []
            for pos in positions_needed:
                index = int(random.triangular(0, len(predicted_points[pos]), 0))
                player_to_add = predicted_points[pos][index][0]
                added_ok = new_squad.add_player(player_to_add, gameweek=transfer_gw)
                if added_ok:
                    added_players.append(player_to_add)
            complete_squad = new_squad.is_complete()
            if not complete_squad:
                # try to avoid getting stuck in a loop
                attempt += 1
                if attempt > max_tries:
                    new_squad = fastcopy(squad)
                    break
                # take those players out again.
                for ap in added_players:
                    removed_ok = new_squad.remove_player(
                        ap.player_id, gameweek=transfer_gw
                    )
                    if not removed_ok:
                        logger.warning("Problem removing %s", ap)
                added_players = []

        # calculate the score
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
            best_pid_out = removed_players
            best_pid_in = [ap.player_id for ap in added_players]
            best_squad = new_squad
            # end of loop over n_iter

    if best_squad is None:
        msg = "Failed to find valid random transfers for squad"
        raise RuntimeError(msg)

    return best_squad, best_pid_out, best_pid_in


class RandomTransferStrategy:
    """Sample random sets of transfers and keep the best squad found."""

    def num_increments(self, move: GameweekMove, num_iterations: int) -> int:  # noqa: ARG002
        return num_iterations

    def propose(self, request: TransferRequest) -> Proposal:
        squad, players_out, players_in = make_random_transfers(
            request.squad,
            request.tag,
            nsubs=request.move.n_transfers,
            gameweeks=request.gameweeks,
            root_gw=request.root_gw,
            num_iter=request.num_iterations,
            on_step=request.progress,
            season=request.season,
            bench_boost_gw=request.bench_boost_gw,
            triple_captain_gw=request.triple_captain_gw,
        )
        return Proposal(squad, players_in, players_out)
