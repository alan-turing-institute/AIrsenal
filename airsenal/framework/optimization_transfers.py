"""
Functions for optimising transfers across multiple gameweeks, including the possibility
of using chips.
"""
import random
from operator import itemgetter

from airsenal.framework.optimization_squad import make_new_squad
from airsenal.framework.optimization_utils import get_discounted_squad_score
from airsenal.framework.utils import (
    CURRENT_SEASON,
    NEXT_GAMEWEEK,
    fastcopy,
    get_predicted_points,
    get_squad_value,
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

    best_score = -1
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
                print(f"Failed to add {p_in[0].name}")
        if not new_squad.is_complete() and verbose:
            print(f"Failed to find a valid replacement for {p_out.player_id}")

    if best_squad is None:
        raise RuntimeError("Failed to find valid single transfer for squad")

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
    best_score = -1
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
        raise RuntimeError("Failed to find valid double transfer for squad")

    return best_squad, best_pid_out, best_pid_in


def make_random_transfers(
    squad,
    tag,
    nsubs=1,
    gw_range=None,
    root_gw=None,
    num_iter=1,
    update_func_and_args=None,
    season=CURRENT_SEASON,
    bench_boost_gw=None,
    triple_captain_gw=None,
):
    """
    choose nsubs random players to sub out, and then select players
    using a triangular PDF to preferentially select the replacements with
    the best expected score to fill their place.
    Do this num_iter times and choose the best total score over gw_range gameweeks.
    """
    best_score = -1
    best_squad = None
    best_pid_out, best_pid_in = [], []
    max_tries = 100
    for _ in range(num_iter):
        if update_func_and_args:
            # call function to update progress bar.
            # this was passed as a tuple (func, increment, pid)
            update_func_and_args[0](update_func_and_args[1], update_func_and_args[2])

        new_squad = fastcopy(squad)

        if not gw_range:
            gw_range = [NEXT_GAMEWEEK]
            root_gw = NEXT_GAMEWEEK

        transfer_gw = min(gw_range)  # the week we're making the transfer
        players_to_remove = []  # this is the index within the squad
        removed_players = []  # this is the player_ids
        # order the players in the squad by predicted_points - least-to-most
        player_list = []
        for p in squad.players:
            p.calc_predicted_points(tag)
            player_list.append((p.player_id, p.predicted_points[tag][gw_range[0]]))
        player_list.sort(key=itemgetter(1), reverse=False)
        while len(players_to_remove) < nsubs:
            index = int(random.triangular(0, len(player_list), 0))
            if index not in players_to_remove:
                players_to_remove.append(index)

        positions_needed = []
        for p in players_to_remove:
            positions_needed.append(squad.players[p].position)
            removed_players.append(squad.players[p].player_id)
            new_squad.remove_player(removed_players[-1], gameweek=transfer_gw)
        predicted_points = {
            pos: get_predicted_points(
                position=pos, gameweek=gw_range, tag=tag, season=season
            )
            for pos in set(positions_needed)
        }
        complete_squad = False
        added_players = []
        attempt = 0
        while not complete_squad:
            # sample with a triangular PDF - preferentially select players near
            # the start
            added_players = []
            for pos in positions_needed:
                index = int(random.triangular(0, len(predicted_points[pos]), 0))
                pid_to_add = predicted_points[pos][index][0]
                added_ok = new_squad.add_player(pid_to_add, gameweek=transfer_gw)
                if added_ok:
                    added_players.append(pid_to_add)
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
                        print(f"Problem removing {ap.name}")
                added_players = []

        # calculate the score
        total_points = get_discounted_squad_score(
            new_squad,
            gw_range,
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
        raise RuntimeError("Failed to find valid random transfers for squad")

    return best_squad, best_pid_out, best_pid_in


def make_best_transfers(
    num_transfers,
    squad,
    tag,
    gameweeks,
    root_gw,
    season,
    num_iter=100,
    update_func_and_args=None,
    algorithm="genetic",
):
    """
    Return a new squad and a dictionary {"in": [player_ids],
                                        "out":[player_ids]}
    """
    transfer_dict = {}
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

    elif num_transfers in ["W", "F"]:
        _out = [p.player_id for p in squad.players]
        budget = get_squad_value(squad)
        if num_transfers == "F":
            gameweeks = [gameweeks[0]]  # for free hit, only need to optimize this week
        new_squad = make_new_squad(
            gameweeks,
            tag=tag,
            budget=budget,
            season=season,
            verbose=0,
            bench_boost_gw=bench_boost_gw,
            triple_captain_gw=triple_captain_gw,
            algorithm=algorithm,
            population_size=num_iter,
            num_iter=num_iter,
            update_func_and_args=update_func_and_args,
        )
        _in = [p.player_id for p in new_squad.players]
        players_in = [p for p in _in if p not in _out]  # remove duplicates
        players_out = [p for p in _out if p not in _in]  # remove duplicates
        transfer_dict = {"in": players_in, "out": players_out}

    else:
        raise RuntimeError(f"Unrecognized value for num_transfers: {num_transfers}")

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
    else:
        return new_squad, transfer_dict, points
