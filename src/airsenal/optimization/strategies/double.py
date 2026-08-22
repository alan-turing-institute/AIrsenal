"""Replace two players, trying every pair in turn."""

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

# 15 players choose 2, ignoring order: 15*14/2 = 105 candidate squads.
NUM_PAIRS = 105


def make_optimum_double_transfer(
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
    If we want to just make two transfers, it's not infeasible to try all
    possibilities in turn.
    We will order the list of potential subs via the sum of expected points
    over a specified range of gameweeks.
    """
    if not gameweek_range:
        gameweek_range = [next_gameweek()]
        root_gw = next_gameweek()

    transfer_gw = min(gameweek_range)  # the week we're making the transfer
    best_score = -1.0
    best_squad = None
    best_pid_out, best_pid_in = [], []
    ordered_player_lists = {
        pos: get_predicted_points(
            gameweek=gameweek_range, position=pos, tag=tag, season=season
        )
        for pos in list(Position.back_to_front())
    }
    for i in range(len(squad.players) - 1):
        positions_needed = []
        pout_1 = squad.players[i]

        new_squad_remove_1 = fastcopy(squad)
        new_squad_remove_1.remove_player(pout_1.player_id, gameweek=transfer_gw)
        for j in range(i + 1, len(squad.players)):
            if update_func_and_args:
                # call function to update progress bar.
                # this was passed as a tuple (func, increment, pid)
                update_func_and_args[0](
                    update_func_and_args[1], update_func_and_args[2]
                )

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
                            gameweek_range,
                            tag,
                            root_gw=root_gw,
                            bench_boost_gw=bench_boost_gw,
                            triple_captain_gw=triple_captain_gw,
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

    def num_increments(self, move: GameweekMove, num_iterations: int) -> int:  # noqa: ARG002
        return NUM_PAIRS

    def propose(self, request: TransferRequest) -> TransferPlan:
        squad, players_out, players_in = make_optimum_double_transfer(
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


@TRANSFER_STRATEGIES.register("double", NoOptions)
def _make(config: NoOptions) -> DoubleTransferStrategy:  # noqa: ARG001
    return DoubleTransferStrategy()
