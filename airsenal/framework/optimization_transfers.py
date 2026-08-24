"""
Functions for optimising transfers across multiple gameweeks, including the possibility
of using chips.
"""

from collections.abc import Callable
from multiprocessing import Process

from airsenal.framework.optimization_squad import SquadOpt, make_new_squad
from airsenal.framework.optimization_utils import get_discounted_squad_score
from airsenal.framework.squad import Squad
from airsenal.framework.utils import (
    CURRENT_SEASON,
    NEXT_GAMEWEEK,
    fastcopy,
    get_predicted_points,
)


def make_optimum_single_transfer(
    squad,
    tag,
    gameweek_range=None,
    root_gw=None,
    season=CURRENT_SEASON,
    update_func_and_args=None,
    bench_boost_gw=None,
    triple_captain_gw=None,
    verbose=False,
):
    """
    If we want to just make one transfer, it's not unfeasible to try all
    possibilities in turn.

    We will order the list of potential transfers via the sum of
    expected points over a specified range of gameweeks.
    """
    if not gameweek_range:
        gameweek_range = [NEXT_GAMEWEEK]
        root_gw = NEXT_GAMEWEEK

    transfer_gw = min(gameweek_range)  # the week we're making the transfer

    best_score = -1.0
    best_squad = None
    best_pid_out, best_pid_in = [], []

    if verbose:
        print("Creating ordered player lists")
    ordered_player_lists = {
        pos: get_predicted_points(
            gameweek=gameweek_range, position=pos, tag=tag, season=season
        )
        for pos in ["GK", "DEF", "MID", "FWD"]
    }
    for p_out in squad.players:
        if update_func_and_args:
            # call function to update progress bar.
            # this was passed as a tuple (func, increment, pid)
            update_func_and_args[0](update_func_and_args[1], update_func_and_args[2])

        new_squad = fastcopy(squad)
        position = p_out.position
        if verbose:
            print(f"Removing player {p_out}")
        new_squad.remove_player(p_out.player_id, gameweek=transfer_gw)
        for p_in in ordered_player_lists[position]:
            if p_in[0].player_id == p_out.player_id:
                continue  # no point in adding the same player back in
            added_ok = new_squad.add_player(p_in[0], gameweek=transfer_gw)
            if added_ok:
                if verbose:
                    print(f"Added player {p_in[0]}")
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
            if verbose:
                print(f"Failed to add {p_in[0]}")
        if not new_squad.is_complete() and verbose:
            print(f"Failed to find a valid replacement for {p_out.player_id}")

    if best_squad is None:
        msg = "Failed to find valid single transfer for squad"
        raise RuntimeError(msg)

    return best_squad, best_pid_out, best_pid_in


def make_optimum_double_transfer(
    squad,
    tag,
    gameweek_range=None,
    root_gw=None,
    season=CURRENT_SEASON,
    update_func_and_args=None,
    bench_boost_gw=None,
    triple_captain_gw=None,
    verbose=False,
):
    """
    If we want to just make two transfers, it's not infeasible to try all
    possibilities in turn.
    We will order the list of potential subs via the sum of expected points
    over a specified range of gameweeks.
    """
    if not gameweek_range:
        gameweek_range = [NEXT_GAMEWEEK]
        root_gw = NEXT_GAMEWEEK

    transfer_gw = min(gameweek_range)  # the week we're making the transfer
    best_score = -1.0
    best_squad = None
    best_pid_out, best_pid_in = [], []
    ordered_player_lists = {
        pos: get_predicted_points(
            gameweek=gameweek_range, position=pos, tag=tag, season=season
        )
        for pos in ["GK", "DEF", "MID", "FWD"]
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
            if verbose:
                print(f"Removing players {i} {j}")
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


def make_optimum_transfers_ga(
    squad,
    tag,
    num_transfers,
    gameweek_range=None,
    root_gw=None,
    season=CURRENT_SEASON,
    num_iter=100,
    update_func_and_args=None,
    bench_boost_gw=None,
    triple_captain_gw=None,
):
    """
    Search for the best squad reachable from `squad` by transferring in at most
    `num_transfers` new players, using the DEAP genetic algorithm (see
    airsenal.framework.optimization_squad.SquadOpt) rather than exhaustively or
    randomly enumerating combinations. This scales to the higher transfer counts
    allowed by saving up free transfers (currently up to 5), where exhaustive
    search over combinations of incoming players is infeasible.
    """
    if not gameweek_range:
        gameweek_range = [NEXT_GAMEWEEK]
        root_gw = NEXT_GAMEWEEK

    if update_func_and_args:
        # call function to update progress bar.
        # this was passed as a tuple (func, increment, pid)
        update_func_and_args[0](update_func_and_args[1], update_func_and_args[2])

    opt = SquadOpt(
        gw_range=gameweek_range,
        tag=tag,
        season=season,
        bench_boost_gw=bench_boost_gw,
        triple_captain_gw=triple_captain_gw,
        base_squad=squad,
        bank=squad.budget,
        max_transfers=num_transfers,
        root_gw=root_gw,
    )
    best_individual, _best_fitness = opt.optimize(
        population_size=num_iter,
        generations=num_iter,
        verbose=False,
    )

    new_squad = opt.decode_individual(best_individual)
    base_ids = {p.player_id for p in squad.players}
    new_ids = {p.player_id for p in new_squad.players}
    players_out = sorted(base_ids - new_ids)
    players_in = sorted(new_ids - base_ids)

    return new_squad, players_out, players_in


def make_best_transfers(
    num_transfers: str | int,
    squad: Squad,
    tag: str,
    gameweeks: list[int],
    root_gw: int,
    season: str,
    num_iter: int = 100,
    update_func_and_args: tuple[Callable, float, Process] | None = None,
) -> tuple[Squad, dict[str, list[int]], float]:
    """
    Return a new squad and a dictionary {"in": [player_ids],
                                        "out":[player_ids]}
    """
    transfer_dict: dict[str, list[int]] = {}
    # deal with triple_captain or free_hit
    triple_captain_gw = None
    bench_boost_gw = None
    if isinstance(num_transfers, str):
        if num_transfers.startswith("T"):
            num_transfers = int(num_transfers[1])
            triple_captain_gw = gameweeks[0]
        elif num_transfers.startswith("B"):
            num_transfers = int(num_transfers[1])
            bench_boost_gw = gameweeks[0]

    if num_transfers == 0:
        # 0 or 'T0' or 'B0' (i.e. zero transfers, possibly with chip)
        new_squad = squad
        transfer_dict = {"in": [], "out": []}
        if update_func_and_args:
            # call function to update progress bar.
            # this was passed as a tuple (func, increment, pid)
            update_func_and_args[0](update_func_and_args[1], update_func_and_args[2])

    elif num_transfers == 1:
        # 1 or 'T1' or 'B1' (i.e. 1 transfer, possibly with chip)
        new_squad, players_out, players_in = make_optimum_single_transfer(
            squad,
            tag,
            gameweeks,
            root_gw,
            season,
            triple_captain_gw=triple_captain_gw,
            bench_boost_gw=bench_boost_gw,
            update_func_and_args=update_func_and_args,
        )
        transfer_dict = {"in": players_in, "out": players_out}

    elif num_transfers == 2:
        # 2 or 'T2' or 'B2' (i.e. 2 transfers, possibly with chip)
        new_squad, players_out, players_in = make_optimum_double_transfer(
            squad,
            tag,
            gameweeks,
            root_gw,
            season,
            triple_captain_gw=triple_captain_gw,
            bench_boost_gw=bench_boost_gw,
            update_func_and_args=update_func_and_args,
        )
        transfer_dict = {"in": players_in, "out": players_out}

    elif isinstance(num_transfers, int) and num_transfers > 2:
        # N or 'TN' or 'BN' (i.e. N transfers, possibly with chip) - GA-based search,
        # scales to the higher transfer counts allowed by saving up free transfers
        # where exhaustive/greedy search over combinations of incoming players isn't.
        new_squad, players_out, players_in = make_optimum_transfers_ga(
            squad,
            tag,
            num_transfers,
            gameweeks,
            root_gw,
            season,
            num_iter=num_iter,
            update_func_and_args=update_func_and_args,
            triple_captain_gw=triple_captain_gw,
            bench_boost_gw=bench_boost_gw,
        )
        transfer_dict = {"in": players_in, "out": players_out}

    elif num_transfers in ["W", "F"]:
        _out = [p.player_id for p in squad.players]
        budget = squad.sale_value(root_gw, use_api=False)
        if num_transfers == "F":
            gameweeks = [gameweeks[0]]  # for free hit, only need to optimize this week
        new_squad = make_new_squad(
            gameweeks,
            tag=tag,
            budget=budget,
            season=season,
            verbose=False,
            bench_boost_gw=bench_boost_gw,
            triple_captain_gw=triple_captain_gw,
            population_size=num_iter,
            generations=num_iter,
        )
        _in = [p.player_id for p in new_squad.players]
        players_in = [p for p in _in if p not in _out]  # remove duplicates
        players_out = [p for p in _out if p not in _in]  # remove duplicates
        transfer_dict = {"in": players_in, "out": players_out}

    else:
        msg = f"Unrecognized value for num_transfers: {num_transfers}"
        raise RuntimeError(msg)

    # get the expected points total for next gameweek
    points = get_discounted_squad_score(
        new_squad,
        [gameweeks[0]],
        tag,
        root_gw=root_gw,
        bench_boost_gw=bench_boost_gw,
        triple_captain_gw=triple_captain_gw,
    )

    if num_transfers == "F":
        # Free Hit changes don't apply to next gameweek, so return the original squad
        return squad, transfer_dict, points
    return new_squad, transfer_dict, points
