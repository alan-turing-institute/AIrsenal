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
    candidate_players_to_remove=[],
    verbose=False,
):
    """
    If we want to just make one transfer, it's not unfeasible to try all
    possibilities in turn, and select the one that gives the highest return
    over a specified number of gameweeks

    Parameters:
       squad: Squad, our starting 15-player squad.
       tag: str, identifier for set of score predictions in the Predictions table to use
       root_gw: int, the gameweek for which this transfer will be applied
       season: str, season under consideration, e.g. "2324" for the 23/24 season.
       update_func_and_args: (function, args) for updating progress bar.
       bench_boost_gw: int, which gameweek to use bench boost (used in score estimation)
       triple_captain_gw: int, gameweek to use triple captain (used in score estimation)
       candidate_players_to_remove: list of CandidatePlayer instances, optional,
           used in place of squad.players if for example we have too many players from
           one team in our squad, and need to remove one of them.

    Returns:
        best_squad: Squad, squad with the best predicted score over next few gameweeks
        best_pid_out: list of int, length 1, player id to remove
        best_pid_in: list of int, length 1, player id to add
    """
    if not gameweek_range:
        gameweek_range = [NEXT_GAMEWEEK]
        root_gw = NEXT_GAMEWEEK

    transfer_gw = min(gameweek_range)  # the week we're making the transfer

    best_score = -1
    best_squad = None
    best_pid_out, best_pid_in = [], []
    # create list of players ordered by how many points they are expected to
    # score over the next few gameweeks
    if verbose:
        print("Creating ordered player lists")
    ordered_player_lists = {
        pos: get_predicted_points(
            gameweek=gameweek_range, position=pos, tag=tag, season=season
        )
        for pos in ["GK", "DEF", "MID", "FWD"]
    }
    # if we have been given a list of players from which to choose our transfer out
    # (e.g. because we have too many from one team in squad), we use that.  Otherwise
    # (i.e. in 99% of cases), we look at all players in our squad.
    if not candidate_players_to_remove:
        candidate_players_to_remove = squad.players
    for p_out in candidate_players_to_remove:
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
    candidate_players_to_remove=[],
    num_compulsory_transfers=0,
    verbose=False,
):
    """
    If we want to just make two transfers, it's not unfeasible to try all
    possibilities in turn.

    Parameters:
       squad: Squad, our starting 15-player squad.
       tag: str, identifier for set of score predictions in the Predictions table to use
       root_gw: int, the gameweek for which this transfer will be applied
       season: str, season under consideration, e.g. "2324" for the 23/24 season.
       update_func_and_args: (function, args) for updating progress bar.
       bench_boost_gw: int, which gameweek to use bench boost (used in score estimation)
       triple_captain_gw: int, gameweek to use triple captain (used in score estimation)
       candidate_players_to_remove: list of CandidatePlayer instances, optional,
           used in place of squad.players if for example we have too many players from
           one team in our squad, and need to remove one of them.
       num_compulsory_transfers: int, should be 0, 1, or 2. If we have non-empty list of
           candidate_players_to_remove, do we need to remove 1 or 2 of them in order to
           make a valid team?   This will determine whether we use that list in place of
           all-players-in-squad, for the outer loop, or both loops.

    Returns:
        best_squad: Squad, squad with the best predicted score over next few gameweeks
        best_pid_out: list of int, length 2, player ids to remove
        best_pid_in: list of int, length 2, player ids to add

    """
    if not gameweek_range:
        gameweek_range = [NEXT_GAMEWEEK]
        root_gw = NEXT_GAMEWEEK

    transfer_gw = min(gameweek_range)  # the week we're making the transfer
    best_score = -1
    best_squad = None
    best_pid_out, best_pid_in = [], []
    # We will order the list of potential subs via the sum of expected points
    # over a specified range of gameweeks.
    ordered_player_lists = {
        pos: get_predicted_points(
            gameweek=gameweek_range, position=pos, tag=tag, season=season
        )
        for pos in ["GK", "DEF", "MID", "FWD"]
    }
    # see whether we should loop over all players in the squad, or over a subset of
    # players, in the inner and outer loops.
    outer_loop_players = squad.players
    inner_loop_players = squad.players
    if num_compulsory_transfers >= 1:
        outer_loop_players = candidate_players_to_remove
        if num_compulsory_transfers >= 2:
            outer_loop_players = candidate_players_to_remove
    for i, pout_1 in enumerate(outer_loop_players):
        positions_needed = []
        new_squad_remove_1 = fastcopy(squad)
        new_squad_remove_1.remove_player(pout_1.player_id, gameweek=transfer_gw)
        for j, pout_2 in enumerate(inner_loop_players):
            # can't remove the same player twice
            if pout_2.player_id == pout_1.player_id:
                continue
            # if we're using the full squad for both inner and outer loops, only
            # need to consider half the matrix
            if num_compulsory_transfers == 0 and i < j:
                continue
            if update_func_and_args:
                # call function to update progress bar.
                # this was passed as a tuple (func, increment, pid)
                update_func_and_args[0](
                    update_func_and_args[1], update_func_and_args[2]
                )
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
    candidate_players_to_remove=[],
):
    """
    choose nsubs random players to sub out, and then select players
    using a triangular PDF to preferentially select the replacements with
    the best expected score to fill their place.
    Do this num_iter times and choose the best total score over gw_range gameweeks.

        Parameters:
       squad: Squad, our starting 15-player squad.
       tag: str, identifier for set of score predictions in the Predictions table to use
       nsubs: int, how many transfers to make
       gw_range: list of int, gameweeks to consider for score predictions.
       root_gw: int, the gameweek for which this transfer will be applied
       num_iter: int, number of iterations tried to find the best squad.
       season: str, season under consideration, e.g. "2324" for the 23/24 season.
       update_func_and_args: (function, args) for updating progress bar.
       bench_boost_gw: int, which gameweek to use bench boost (used in score estimation)
       triple_captain_gw: int, gameweek to use triple captain (used in score estimation)
       candidate_players_to_remove: list of CandidatePlayer instances, optional,
           used in place of squad.players if for example we have too many players from
           one team in our squad, and need to remove one of them.

    Returns:
        best_squad: Squad, squad with the best predicted score over next few gameweeks
        best_pid_out: list of int, length nsubs, player ids to remove
        best_pid_in: list of int, length nsubs, player ids to add
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
        if not candidate_players_to_remove:
            candidate_players_to_remove = squad.players
        for p in candidate_players_to_remove:
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

    Parameters:
        num_transfers: int, or str, how many transfers to make (given by strategy).
           Typically this will be 0, 1, or 2 indicating 0, 1, or 2 transfers, but, could
           be e.g. "T2" indicating triple_captain chip and 2 transfers,
           or "W" or "F" would indicate making a whole new squad after wildcard or
           free hit chip.
        squad: Squad, starting squad.
        tag: str, identifier for retrieving predictions from the database.
        gameweeks: list of int, gameweek range to use for points estimation
        root_gw: int, gameweek for which to make transfer(s).
        season: str, season under consideration, e.g. "2324" for the 2023/24 season.
        num_iter: int, number of iterations of trying to find the best squad,
           if creating a new squad from scratch.
        update_func_and_args: (func, args), progress bar function.
        algorithm: str, whether to use genetic algorithm or classic algorithm when
           making a new squad from scratch.
    """
    # prepare the dict containing lists of player_ids to transfer in and out.
    transfer_dict = {"in": [], "out": []}
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
        # for wildcard and free hit, make a whole new squad
    if num_transfers in ["W", "F"]:
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
        transfer_dict["in"] += players_in
        transfer_dict["out"] += players_out
    else:
        # not a wildcard or free hit - modify our existing squad.
        # First see if we need to do any compulsory transfers e.g. if we have
        # >3 players from the same team, after the (real) transfer window
        candidate_players_to_remove, num_players_to_remove = find_compulsory_transfers(
            squad
        )

        if num_transfers == 0:
            # 0 or 'T0' or 'B0' (i.e. zero transfers, possibly with chip)
            new_squad = squad
            if update_func_and_args:
                # call function to update progress bar.
                # this was passed as a tuple (func, increment, pid)
                update_func_and_args[0](
                    update_func_and_args[1], update_func_and_args[2]
                )

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
                candidate_players_to_remove=candidate_players_to_remove,
            )
            transfer_dict["in"] += players_in
            transfer_dict["out"] += players_out

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
                candidate_players_to_remove=candidate_players_to_remove,
                num_compulsory_transfers=num_players_to_remove,
            )
            transfer_dict["in"] += players_in
            transfer_dict["out"] += players_out

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


def find_compulsory_transfers(squad):
    """
    It can happen that during the real transfer window, a player will change team
    resulting in our FPL squad having >3 players from the same team.
    In this case, we need to transfer player(s) out to make a legal squad.
    This function will find out how many players we need to take out, and
    a list of player_ids from which to choose that number of players to remove.

    Returns:
        cand_players_to_remove [player_id:int,...], num_players_to_remove
    """
    players_per_team, is_legal = squad.players_per_team()
    if is_legal:
        return [], 0

    # count, and make list of players where there are >3 players in same team
    num_players_to_remove = 0
    candidate_players_to_remove = []
    for v in players_per_team.values():
        if len(v) > 3:
            num_players_to_remove += len(v) - 3
            candidate_players_to_remove += v
    return candidate_players_to_remove, num_players_to_remove
