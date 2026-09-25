"""
Replace three or more players by sampling.

There are too many combinations to enumerate once more than two players change,
so this samples random swaps and keeps the best squad it finds.
"""

import random
from functools import partial
from operator import itemgetter
from typing import TYPE_CHECKING

from airsenal.core.copy import fastcopy
from airsenal.core.logging import get_logger
from airsenal.db.queries.predictions import get_predicted_points
from airsenal.optimization.protocols import Proposal, TransferRequest
from airsenal.optimization.squad_score import get_discounted_squad_score
from airsenal.squad.squad import Squad

if TYPE_CHECKING:
    from airsenal.db.models import Player

logger = get_logger(__name__)


def make_random_transfers(
    request: TransferRequest,
) -> tuple[Squad, list[int], list[int]]:
    """
    Sample `request.num_iterations` sets of transfers at random and keep the best.

    Each iteration drops `request.move.n_transfers` players and fills their places
    from the candidates for those positions. Both draws use a triangular
    distribution with its mode at index 0, so the front of each list comes up most
    often. Candidates are priced as at `request.root_gameweek`.
    """
    squad = request.squad
    tag = request.tag
    gameweeks = request.gameweeks
    root_gameweek = request.root_gameweek
    nsubs = request.move.n_transfers
    score = partial(
        get_discounted_squad_score,
        gameweeks=gameweeks,
        tag=tag,
        root_gameweek=root_gameweek,
        bench_boost_gameweek=request.bench_boost_gameweek,
        triple_captain_gameweek=request.triple_captain_gameweek,
        sub_weights=request.scoring.sub_weights,
    )

    # order the players in the squad by predicted_points - least-to-most
    player_list: list[tuple[int, float]] = []
    for p in squad.players:
        p.calc_predicted_points(tag)
        player_list.append((p.player_id, p.predicted_points[tag][gameweeks[0]]))
    player_list.sort(key=itemgetter(1), reverse=False)
    predicted_points: dict[str, list[tuple[Player, float]]] = {}

    best_score = -1.0
    best_squad = None
    best_pid_out, best_pid_in = [], []
    max_tries = 100
    for _ in range(request.num_iterations):
        request.advance_progress()

        new_squad = fastcopy(squad)

        # indices into player_list, which is sorted; not into squad.players
        players_to_remove: list[int] = []
        removed_players: list[int] = []  # this is the player_ids
        while len(players_to_remove) < nsubs:
            index = int(random.triangular(0, len(player_list), 0))
            if index not in players_to_remove:
                players_to_remove.append(index)

        positions_needed = []
        # The triangular draw above prefers low indices, which after the sort are
        # the worst players.
        for list_index in players_to_remove:
            player_id = player_list[list_index][0]
            positions_needed.append(squad.get_player_from_id(player_id).position)
            removed_players.append(player_id)
            new_squad.remove_player(player_id, gameweek=root_gameweek)
        for pos in positions_needed:
            if pos not in predicted_points:
                predicted_points[pos] = get_predicted_points(
                    position=pos, gameweeks=gameweeks, tag=tag, season=request.season
                )
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
                added_ok = new_squad.add_player(player_to_add, gameweek=root_gameweek)
                if added_ok:
                    added_players.append(player_to_add)
            complete_squad = new_squad.is_complete()
            if not complete_squad:
                # try to avoid getting stuck in a loop
                attempt += 1
                if attempt > max_tries:
                    new_squad = fastcopy(squad)
                    removed_players, added_players = [], []
                    break
                # take those players out again.
                for ap in added_players:
                    removed_ok = new_squad.remove_player(
                        ap.player_id, gameweek=root_gameweek
                    )
                    if not removed_ok:
                        logger.warning("Problem removing %s", ap)
                added_players = []

        total_points = score(new_squad)
        if total_points > best_score:
            best_score = total_points
            best_pid_out = removed_players
            best_pid_in = [ap.player_id for ap in added_players]
            best_squad = new_squad

    if best_squad is None:
        msg = "Failed to find valid random transfers for squad"
        raise RuntimeError(msg)

    return best_squad, best_pid_out, best_pid_in


class RandomTransferStrategy:
    """Sample random sets of transfers and keep the best squad found."""

    def num_increments(self, request: TransferRequest) -> int:
        return request.num_iterations

    def propose(self, request: TransferRequest) -> Proposal:
        squad, players_out, players_in = make_random_transfers(request)
        return Proposal(squad, players_in, players_out)
