"""
functions to optimize the transfers for N weeks ahead
"""

import random
from datetime import datetime
from operator import itemgetter

from airsenal.framework.schema import TransferSuggestion, Transaction
from airsenal.framework.squad import Squad, TOTAL_PER_POSITION
from airsenal.framework.player import CandidatePlayer
from airsenal.framework.utils import (
    session,
    NEXT_GAMEWEEK,
    CURRENT_SEASON,
    get_predicted_points,
    fastcopy,
    get_team_value
)

positions = ["FWD", "MID", "DEF", "GK"]  # front-to-back


def generate_transfer_strategies(
        gw_ahead,
        free_transfers=1,
        max_total_hit=None,
        allow_wildcard=False,
        allow_free_hit=False,
        allow_bench_boost=False,
        allow_triple_captain=False,
        allow_unused_transfers=True,
        next_gw=NEXT_GAMEWEEK,
):
    """
    Generate possible transfer and chip strategies for gw_ahead gameweeks ahead,
    subject to:
    * Make a maximum of 3 transfers each gameweek.
    * If a chip is played only allow 0 or 1 transfers to be played or/in combination
      with a chip.
    * Each chip only allowed once.
    * Spend a max of max_total_hit points on transfers across whole period.
    • Start with free_transfers free transfers.

    Return all possible transfer sequences along with the total points hit and the
    number of free transfers available for the next gameweek, i.e. return value is a
    list of tuples:
        [({gw:ntransfer, ...},points_hit, free_transfers), ... ]
    """

    # if not using a chip, consider strategies taking up to 3 transfers each week
    if not (
            allow_wildcard or allow_free_hit or allow_bench_boost or allow_triple_captain
    ):
        possibilities = list(range(4))
    # if using a chip, only consider strategies taking up to 1 transfer (or a chip)
    else:
        possibilities = [0, 1]
        if allow_wildcard:
            possibilities.append("W")
        if allow_free_hit:
            possibilities.append("F")
        if allow_bench_boost:
            # bench boost and no or one transfer
            possibilities.append("B0")
            possibilities.append("B1")
        if allow_triple_captain:
            # triple captain and no or one transfer
            possibilities.append("T0")
            possibilities.append("T1")

    # first week strategies
    strategies = []
    for n_transfers in possibilities:
        strat = {next_gw: n_transfers}
        if n_transfers in ["W", "F"]:
            hit_so_far = 0
            new_free_transfers = 1
        else:
            if n_transfers in ["B0", "B1", "T0", "T1"]:
                n_transfers = int(n_transfers[1])
            hit_so_far = 4 * max(0, n_transfers - free_transfers)
            new_free_transfers = 2 if n_transfers < free_transfers else 1

        # only keep strategies that:
        # - don't exceed max_total_hit, if given
        # - don't leave transfers unused, if allow_unused_transfers is True
        # baseline of no transfers added back in at end if removed by this
        if (max_total_hit is None or hit_so_far <= max_total_hit) and not (
                not allow_unused_transfers and free_transfers == 2 and n_transfers == 0
        ):
            strategies.append((strat, hit_so_far, new_free_transfers))

    # subsequent weeks strategies
    for gw in range(next_gw + 1, next_gw + gw_ahead):
        new_strategies = []
        for s in strategies:
            # s is a tuple ( {gw: num_transfer, ...} , points_hit, free_transfers)
            hit_so_far = s[1]
            free_transfers = s[2]

            if not (
                    allow_wildcard
                    or allow_free_hit
                    or allow_bench_boost
                    or allow_triple_captain
            ):
                possibilities = list(range(4))
            else:
                already_used_wildcard = "W" in s[0].values()
                already_used_free_hit = "F" in s[0].values()
                already_used_bench_boost = (
                        "B0" in s[0].values() or "B1" in s[0].values()
                )
                already_used_triple_captain = (
                        "T0" in s[0].values() or "T1" in s[0].values()
                )

                possibilities = [0, 1]
                if allow_wildcard and not already_used_wildcard:
                    possibilities.append("W")
                if allow_free_hit and not already_used_free_hit:
                    possibilities.append("F")
                if allow_bench_boost and not already_used_bench_boost:
                    possibilities.append("B0")
                    possibilities.append("B1")
                if allow_triple_captain and not already_used_triple_captain:
                    possibilities.append("T0")
                    possibilities.append("T1")

            for n_transfers in possibilities:
                # make a copy of the strategy up to this point, then add on this gw
                new_dict = {k: v for k, v in s[0].items()}
                new_dict[gw] = n_transfers

                # determine total pts hit, and free transfers for the following gw
                if n_transfers in ["W", "F"]:
                    # no hit if using wildcard/free hit, plus lose any saved free transfers
                    new_hit = hit_so_far
                    new_free_transfers = 1
                else:
                    if n_transfers in ["B0", "B1", "T0", "T1"]:
                        n_transfers = int(n_transfers[1])
                    new_hit = hit_so_far + 4 * max(n_transfers - free_transfers, 0)
                    new_free_transfers = 2 if n_transfers < free_transfers else 1

                # only keep strategies that:
                # - don't exceed max_total_hit, if given
                # - don't leave transfers unused, if allow_unused_transfers is True
                if (max_total_hit is None or new_hit <= max_total_hit) and not (
                        not allow_unused_transfers
                        and free_transfers == 2
                        and n_transfers == 0
                ):
                    new_strategies.append((new_dict, new_hit, new_free_transfers))

        strategies = new_strategies

    # if allow_unused_transfers is False baseline of no transfers will be removed above,
    # add it back in here.
    if not allow_unused_transfers:
        baseline_dict = ({gw: 0 for gw in range(next_gw, next_gw + gw_ahead)}, 0, 2)
        strategies.insert(0, baseline_dict)

    return strategies


def get_starting_squad():
    """
    use the transactions table in the db
    """
    s = Squad()
    transactions = session.query(Transaction).order_by(Transaction.id).all()
    for trans in transactions:
        if trans.bought_or_sold == -1:
            s.remove_player(trans.player_id, price=trans.price)
        else:
            ## within an individual transfer we can violate the budget and team constraints,
            ## as long as the final team for that gameweek obeys them
            s.add_player(
                trans.player_id,
                price=trans.price,
                season=trans.season,
                gameweek=trans.gameweek,
                check_budget=False,
                check_team=False,
            )
    return s


def get_baseline_prediction(gw_ahead, tag):
    """
    use current squad, and count potential score
    also return a cumulative total per gw, so we can abort if it
    looks like we're doing too badly.
    """
    squad = get_starting_squad()
    total = 0.0
    cum_total_per_gw = {}
    next_gw = NEXT_GAMEWEEK
    gameweeks = list(range(next_gw, next_gw + gw_ahead))
    for gw in gameweeks:
        score = squad.get_expected_points(gw, tag)
        cum_total_per_gw[gw] = total + score
        total += score
    return total, cum_total_per_gw


def make_optimum_transfer(
        squad,
        tag,
        gameweek_range=None,
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
        gameweek_range = [NEXT_GAMEWEEK]

    transfer_gw = min(gameweek_range)  # the week we're making the transfer
    best_score = 0.0
    best_pid_out, best_pid_in = 0, 0
    ordered_player_lists = {}
    for pos in ["GK", "DEF", "MID", "FWD"]:
        ordered_player_lists[pos] = get_predicted_points(
            gameweek=gameweek_range, position=pos, tag=tag
        )
    for p_out in squad.players:
        if update_func_and_args:
            ## call function to update progress bar.
            ## this was passed as a tuple (func, increment, pid)
            update_func_and_args[0](update_func_and_args[1], update_func_and_args[2])

        new_squad = fastcopy(squad)
        position = p_out.position
        new_squad.remove_player(p_out.player_id, season=season, gameweek=transfer_gw)
        for p_in in ordered_player_lists[position]:
            if p_in[0].player_id == p_out.player_id:
                continue  # no point in adding the same player back in
            added_ok = new_squad.add_player(p_in[0], season=season, gameweek=transfer_gw)
            if added_ok:
                break
        total_points = 0.0
        for gw in gameweek_range:
            if gw == bench_boost_gw:
                total_points += new_squad.get_expected_points(gw, tag, bench_boost=True)
            elif gw == triple_captain_gw:
                total_points += new_squad.get_expected_points(
                    gw, tag, triple_captain=True
                )
            else:
                total_points += new_squad.get_expected_points(gw, tag)
        if total_points > best_score:
            best_score = total_points
            best_pid_out = p_out.player_id
            best_pid_in = p_in[0].player_id
            best_squad = new_squad
    return best_squad, [best_pid_out], [best_pid_in]


def make_optimum_double_transfer(
        squad,
        tag,
        gameweek_range=None,
        season=CURRENT_SEASON,
        update_func_and_args=None,
        verbose=False,
        bench_boost_gw=None,
        triple_captain_gw=None,
):
    """
    If we want to just make two transfers, it's not unfeasible to try all
    possibilities in turn.
    We will order the list of potential subs via the sum of expected points
    over a specified range of gameweeks.
    """
    if not gameweek_range:
        gameweek_range = [NEXT_GAMEWEEK]

    transfer_gw = min(gameweek_range)  # the week we're making the transfer
    best_score = 0.0
    best_pid_out, best_pid_in = 0, 0
    ordered_player_lists = {}
    for pos in ["GK", "DEF", "MID", "FWD"]:
        ordered_player_lists[pos] = get_predicted_points(
            gameweek=gameweek_range, position=pos, tag=tag
        )

    for i in range(len(squad.players) - 1):
        positions_needed = []
        pout_1 = squad.players[i]

        new_squad_remove_1 = fastcopy(squad)
        new_squad_remove_1.remove_player(

            pout_1.player_id, season=season, gameweek=transfer_gw
        )
        for j in range(i + 1, len(squad.players)):
            if update_func_and_args:
                ## call function to update progress bar.
                ## this was passed as a tuple (func, increment, pid)
                update_func_and_args[0](
                    update_func_and_args[1], update_func_and_args[2]
                )

            pout_2 = squad.players[j]
            new_squad_remove_2 = fastcopy(new_squad_remove_1)
            new_squad_remove_2.remove_player(
                pout_2.player_id, season=season, gameweek=transfer_gw
            )
            if verbose:
                print("Removing players {} {}".format(i, j))
            ## what positions do we need to fill?
            positions_needed = [pout_1.position, pout_2.position]

            # now loop over lists of players and add players back in
            for pin_1 in ordered_player_lists[positions_needed[0]]:
                if (
                        pin_1[0].player_id == pout_1.player_id
                        or pin_1[0].player_id == pout_2.player_id
                ):
                    continue  ## no point in adding same player back in
                new_squad_add_1 = fastcopy(new_squad_remove_2)
                added_1_ok = new_squad_add_1.add_player(
                    pin_1[0], season=season, gameweek=transfer_gw
                )
                if not added_1_ok:
                    continue
                for pin_2 in ordered_player_lists[positions_needed[1]]:
                    new_squad_add_2 = fastcopy(new_squad_add_1)
                    if (
                            pin_2[0] == pin_1[0]
                            or pin_2[0].player_id == pout_1.player_id
                            or pin_2[0].player_id == pout_2.player_id
                    ):
                        continue  ## no point in adding same player back in
                    added_2_ok = new_squad_add_2.add_player(
                        pin_2[0], season=season, gameweek=transfer_gw
                    )
                    if added_2_ok:
                        # calculate the score
                        total_points = 0.0
                        for gw in gameweek_range:
                            if gw == bench_boost_gw:
                                total_points += new_squad_add_2.get_expected_points(
                                    gw, tag, bench_boost=True
                                )
                            elif gw == triple_captain_gw:
                                total_points += new_squad_add_2.get_expected_points(
                                    gw, tag, triple_captain=True
                                )
                            else:
                                total_points += new_squad_add_2.get_expected_points(
                                    gw, tag
                                )
                        if total_points > best_score:
                            best_score = total_points
                            best_pid_out = [pout_1.player_id, pout_2.player_id]
                            best_pid_in = [pin_1[0].player_id, pin_2[0].player_id]
                            best_squad = new_squad_add_2
                        break

    return best_squad, best_pid_out, best_pid_in


def make_random_transfers(
        squad,
        tag,
        nsubs=1,
        gw_range=None,
        num_iter=1,
        update_func_and_args=None,
        season=CURRENT_SEASON,
        bench_boost_gw=None,
        triple_captain_gw=None,
):
    """
    choose nsubs random players to sub out, and then select players
    using a triangular PDF to preferentially select  the replacements with
    the best expected score to fill their place.
    Do this num_iter times and choose the best total score over gw_range gameweeks.
    """
    best_score = 0.0
    best_squad = None
    best_pid_out = []
    best_pid_in = []
    max_tries = 100
    for i in range(num_iter):
        if update_func_and_args:
            ## call function to update progress bar.
            ## this was passed as a tuple (func, increment, pid)
            update_func_and_args[0](update_func_and_args[1], update_func_and_args[2])

        new_squad = fastcopy(squad)

        if not gw_range:
            gw_range = [NEXT_GAMEWEEK]

        transfer_gw = min(gw_range)  # the week we're making the transfer
        players_to_remove = []  # this is the index within the squad
        removed_players = []  # this is the player_ids
        ## order the players in the squad by predicted_points - least-to-most
        player_list = []
        for p in squad.players:
            p.calc_predicted_points(tag)
            player_list.append((p.player_id, p.predicted_points[tag][gw_range[0]]))
        player_list.sort(key=itemgetter(1), reverse=False)
        while len(players_to_remove) < nsubs:
            index = int(random.triangular(0, len(player_list), 0))
            if not index in players_to_remove:
                players_to_remove.append(index)

        positions_needed = []
        for p in players_to_remove:
            positions_needed.append(squad.players[p].position)
            removed_players.append(squad.players[p].player_id)
            new_squad.remove_player(
                removed_players[-1], season=season, gameweek=transfer_gw
            )
        budget = new_squad.budget
        predicted_points = {}
        for pos in set(positions_needed):
            predicted_points[pos] = get_predicted_points(
                position=pos, gameweek=gw_range, tag=tag
            )
        complete_squad = False
        added_players = []
        attempt = 0
        while not complete_squad:
            ## sample with a triangular PDF - preferentially select players near
            ## the start
            added_players = []
            for pos in positions_needed:
                index = int(random.triangular(0, len(predicted_points[pos]), 0))
                pid_to_add = predicted_points[pos][index][0]
                added_ok = new_squad.add_player(
                    pid_to_add, season=season, gameweek=transfer_gw
                )
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
                        ap.player_id, season=season, gameweek=transfer_gw
                    )
                    if not removed_ok:
                        print("Problem removing {}".format(ap.name))
                added_players = []

        ## calculate the score
        total_points = 0.0
        for gw in gw_range:
            if gw == bench_boost_gw:
                total_points += new_squad.get_expected_points(gw, tag, bench_boost=True)
            elif gw == triple_captain_gw:
                total_points += new_squad.get_expected_points(
                    gw, tag, triple_captain=True
                )
            else:
                total_points += new_squad.get_expected_points(gw, tag)
        if total_points > best_score:
            best_score = total_points
            best_pid_out = removed_players
            best_pid_in = [ap.player_id for ap in added_players]
            best_squad = new_squad
        ## end of loop over n_iter
    return best_squad, best_pid_out, best_pid_in


def make_new_squad(
        budget,
        num_iterations,
        tag,
        gw_range,
        season=CURRENT_SEASON,
        session=None,
        update_func_and_args=None,
        verbose=False,
        bench_boost_gw=None,
        triple_captain_gw=None,
):
    """
    Make a squad from scratch, i.e. for gameweek 1, or for wildcard, or free hit.
    """
    transfer_gw = min(gw_range)  # the gw we're making the new squad
    best_score = 0.0
    best_squad = None

    for iteration in range(num_iterations):
        if verbose:
            print("Choosing new squad: iteration {}".format(iteration))
        if update_func_and_args:
            ## call function to update progress bar.
            ## this was passed as a tuple (func, increment, pid)
            update_func_and_args[0](update_func_and_args[1], update_func_and_args[2])
        predicted_points = {}
        t = Squad(budget)
        # first iteration - fill up from the front
        for pos in positions:
            predicted_points[pos] = get_predicted_points(
                gameweek=gw_range, position=pos, tag=tag, season=season
            )
            for pp in predicted_points[pos]:
                t.add_player(pp[0], season=season, gameweek=transfer_gw)
                if t.num_position[pos] == TOTAL_PER_POSITION[pos]:
                    break

        # presumably we didn't get a complete squad now
        excluded_player_ids = []
        while not t.is_complete():
            # randomly swap out a player and replace with a cheaper one in the
            # same position
            player_to_remove = t.players[random.randint(0, len(t.players) - 1)]
            remove_cost = player_to_remove.purchase_price
            remove_position = player_to_remove.position
            t.remove_player(
                player_to_remove.player_id, season=season, gameweek=transfer_gw
            )
            excluded_player_ids.append(player_to_remove.player_id)
            for pp in predicted_points[player_to_remove.position]:
                if (
                        not pp[0] in excluded_player_ids
                ) or random.random() < 0.3:  # some chance to put player back
                    cp = CandidatePlayer(pp[0], gameweek=transfer_gw, season=season)
                    if cp.purchase_price >= remove_cost:
                        continue
                    else:
                        t.add_player(pp[0], season=season, gameweek=transfer_gw)
            # now try again to fill up the rest of the squad
            num_missing_per_position = {}

            for pos in positions:
                num_missing = TOTAL_PER_POSITION[pos] - t.num_position[pos]
                if num_missing == 0:
                    continue
                for pp in predicted_points[pos]:
                    if pp[0] in excluded_player_ids:
                        continue
                    t.add_player(pp[0], season=season, gameweek=transfer_gw)
                    if t.num_position[pos] == TOTAL_PER_POSITION[pos]:
                        break
        # we have a complete squad
        score = 0.0
        for gw in gw_range:
            if gw == bench_boost_gw:
                score += t.get_expected_points(gw, tag, bench_boost=True)
            elif gw == triple_captain_gw:
                score += t.get_expected_points(gw, tag, triple_captain=True)
            else:
                score += t.get_expected_points(gw, tag)
        if score > best_score:
            best_score = score
            best_squad = t

    if verbose:
        print("====================================\n")
        print(best_squad)
        print(best_score)
    return best_squad


def apply_strategy(
        strat, tag, baseline_dict=None, num_iter=1, update_func_and_args=None, verbose=False
):
    """
    apply a set of transfers over a number of gameweeks, and
    total up the score, taking into account points hits.
    strat is a tuple, with the first element being the
    dictionary {gw:ntransfers,...} and the second element being
    the total points hit.
    """
    sid = make_strategy_id(strat)
    starting_squad = get_starting_squad()
    if verbose:
        print("Trying strategy {}".format(strat))
    best_score = 0
    best_strategy_output = {}
    gameweeks = sorted(strat[0].keys())  # go through gameweeks in order
    if verbose:
        print(" --> doing strategy {}".format(sid))
    strategy_output = {
        "total_score": -1 * strat[1],  # points hit from this strategy
        "points_per_gw": {},
        "players_in": {},
        "players_out": {},
        "cards_played": {},
    }
    new_squad = fastcopy(starting_squad)
    ## If we use "free hit" card, we need to remember the team from the week before it
    squad_before_free_hit = None

    # determine if bench boost or triple captain used in this strategy
    bench_boost_gw = None
    triple_captain_gw = None
    for gw, n_transfers in strat[0].items():
        if n_transfers in ["B0", "B1"]:
            bench_boost_gw = gw
        elif n_transfers in ["T0", "T1"]:
            triple_captain_gw = gw

    for igw, gw in enumerate(gameweeks):
        ## how many gameweeks ahead should we look at for the purpose of estimating points?
        gw_range = gameweeks[igw:]  # range of gameweeks to end of window

        ## if we used a free hit in the previous gw, we will have stored the previous squad, so
        ## we go back to that one now.

        if squad_before_free_hit:
            new_squad = fastcopy(squad_before_free_hit)
            squad_before_free_hit = None

        ## process this gameweek
        if strat[0][gw] == 0:  # no transfers that gameweek
            rp, ap = [], []  ## lists of removed-players, added-players

        elif strat[0][gw] == 1:  # one transfer - choose optimum
            new_squad, rp, ap = make_optimum_transfer(
                new_squad,
                tag,
                gw_range,
                update_func_and_args=update_func_and_args,
                bench_boost_gw=bench_boost_gw,
                triple_captain_gw=triple_captain_gw,
            )
        elif strat[0][gw] == 2:
            ## two transfers - choose optimum
            new_squad, rp, ap = make_optimum_double_transfer(
                new_squad,
                tag,
                gw_range,
                update_func_and_args=update_func_and_args,
                bench_boost_gw=bench_boost_gw,
                triple_captain_gw=triple_captain_gw,
            )
        elif strat[0][gw] == "W":  ## wildcard - a whole new squad!
            rp = [p.player_id for p in new_squad.players]
            budget = get_team_value(new_squad)
            new_squad = make_new_squad(
                budget,
                num_iter,
                tag,
                gw_range,
                update_func_and_args=update_func_and_args,
                bench_boost_gw=bench_boost_gw,
                triple_captain_gw=triple_captain_gw,
            )

            ap = [p.player_id for p in new_squad.players]

        elif strat[0][gw] == "F":  ## free hit - a whole new squad!
            ## remember the starting team (so we can revert to it later)
            squad_before_free_hit = fastcopy(new_squad)
            ## now make a new squad for this gw, as is done for wildcard
            new_team = [p.player_id for p in new_squad.players]
            budget = get_team_value(new_team)
            new_squad = make_new_squad(
                budget,
                num_iter,
                tag,
                [gw],  # free hit should be optimised for this gameweek only
                update_func_and_args=update_func_and_args,
                bench_boost_gw=bench_boost_gw,
                triple_captain_gw=triple_captain_gw,
            )
            ap = [p.player_id for p in new_squad.players]

        elif strat[0][gw] in ["B0", "T0"]:  # bench boost/triple captain and no transfer
            rp, ap = [], []  ## lists of removed-players, added-players

        elif strat[0][gw] in [
            "B1",
            "T1",
        ]:  # bench boost/triple captain and one transfer
            new_squad, rp, ap = make_optimum_transfer(
                new_squad,
                tag,
                gw_range,
                update_func_and_args=update_func_and_args,
                bench_boost_gw=bench_boost_gw,
                triple_captain_gw=triple_captain_gw,
            )

        else:  # choose randomly
            new_squad, rp, ap = make_random_transfers(
                new_squad,
                tag,
                strat[0][gw],
                gw_range,
                num_iter=num_iter,
                update_func_and_args=update_func_and_args,
                bench_boost_gw=bench_boost_gw,
                triple_captain_gw=triple_captain_gw,
            )

        if gw == bench_boost_gw:
            score = new_squad.get_expected_points(gw, tag, bench_boost=True)
        elif gw == triple_captain_gw:
            score = new_squad.get_expected_points(gw, tag, triple_captain=True)
        else:
            score = new_squad.get_expected_points(gw, tag)

        ## if we're ever >5 points below the baseline, bail out!
        strategy_output["total_score"] += score
        if baseline_dict and baseline_dict[gw] - strategy_output["total_score"] > 5:
            break
        strategy_output["points_per_gw"][gw] = score
        # record whether we're playing a chip this gameweek
        # only first character to remove number transfers in case of bench boost or
        # triple captain ("B" and "T", not "B0", "B1", "T0", "T1")
        strategy_output["cards_played"][gw] = (
            strat[0][gw][0] if isinstance(strat[0][gw], str) else None
        )
        strategy_output["players_in"][gw] = ap
        strategy_output["players_out"][gw] = rp
        ## end of loop over gameweeks
    if strategy_output["total_score"] > best_score:
        best_score = strategy_output["total_score"]
        best_strategy_output = strategy_output
    if verbose:
        print("Total score: {}".format(best_strategy_output["total_score"]))
    return best_strategy_output


def fill_suggestion_table(baseline_score, best_strat, season):
    """
    Fill the optimized strategy into the table
    """
    timestamp = str(datetime.now())
    best_score = best_strat["total_score"]
    points_gain = best_score - baseline_score
    for in_or_out in [("players_out", -1), ("players_in", 1)]:
        for gameweek, players in best_strat[in_or_out[0]].items():
            for player in players:
                ts = TransferSuggestion()
                ts.player_id = player
                ts.in_or_out = in_or_out[1]
                ts.gameweek = gameweek
                ts.points_gain = points_gain
                ts.timestamp = timestamp
                ts.season = season
                session.add(ts)
    session.commit()


def strategy_involves_N_or_more_transfers_in_gw(strategy, N):
    """
    Quick function to see if we need to do multiple iterations
    for a strategy, or if the result is deterministic
    (0 or 1 transfer for each gameweek).
    """
    strat_dict = strategy[0]
    for v in strat_dict.values():
        if isinstance(v, int) and v >= N:
            return True
    return False


def make_strategy_id(strategy):
    """
    Return a string that will identify a strategy - just concatenate
    the numbers of transfers per gameweek.
    """
    strat_id = ",".join([str(nt) for nt in strategy[0].values()])
    return strat_id
