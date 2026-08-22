"""Replace one player, trying every possibility in turn."""

from airsenal.core.copy import fastcopy
from airsenal.core.enums import Position
from airsenal.core.logging import get_logger
from airsenal.db.queries.gameweeks import next_gameweek
from airsenal.db.queries.predictions import get_predicted_points
from airsenal.domain.season import CURRENT_SEASON
from airsenal.optimization.moves import GameweekMove
from airsenal.optimization.protocols import TransferPlan, TransferRequest
from airsenal.optimization.strategies.registry import TRANSFER_STRATEGIES, NoOptions
from airsenal.optimization.utils import get_discounted_squad_score

logger = get_logger(__name__)

# 15 players in a squad, so 15 candidate squads to score.
SQUAD_SIZE = 15


def make_optimum_single_transfer(
    squad,
    tag,
    gameweek_range=None,
    root_gw=None,
    season=CURRENT_SEASON,
    update_func_and_args=None,
    bench_boost_gw=None,
    triple_captain_gw=None,
):
    """
    If we want to just make one transfer, it's not unfeasible to try all
    possibilities in turn.

    We will order the list of potential transfers via the sum of
    expected points over a specified range of gameweeks.
    """
    if not gameweek_range:
        gameweek_range = [next_gameweek()]
        root_gw = next_gameweek()

    transfer_gw = min(gameweek_range)  # the week we're making the transfer

    best_score = -1.0
    best_squad = None
    best_pid_out, best_pid_in = [], []

    logger.debug("Creating ordered player lists")
    ordered_player_lists = {
        pos: get_predicted_points(
            gameweek=gameweek_range, position=pos, tag=tag, season=season
        )
        for pos in list(Position.back_to_front())
    }
    for p_out in squad.players:
        if update_func_and_args:
            # call function to update progress bar.
            # this was passed as a tuple (func, increment, pid)
            update_func_and_args[0](update_func_and_args[1], update_func_and_args[2])

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
                    gameweek_range,
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

    def propose(self, request: TransferRequest) -> TransferPlan:
        squad, players_out, players_in = make_optimum_single_transfer(
            request.squad,
            request.tag,
            request.gameweeks,
            request.root_gw,
            request.season,
            update_func_and_args=request.progress,
            bench_boost_gw=request.bench_boost_gw,
            triple_captain_gw=request.triple_captain_gw,
        )
        return TransferPlan(squad, players_in, players_out)


@TRANSFER_STRATEGIES.register("single", NoOptions)
def _make(config: NoOptions) -> SingleTransferStrategy:  # noqa: ARG001
    return SingleTransferStrategy()
