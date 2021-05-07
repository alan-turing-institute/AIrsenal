"""
functions to optimize the transfers for N weeks ahead
"""

import random
from datetime import datetime
from operator import itemgetter

from airsenal.framework.schema import (
    TransferSuggestion,
    Transaction,
    PlayerPrediction,
    Fixture,
)
from airsenal.framework.squad import Squad, TOTAL_PER_POSITION
from airsenal.framework.player import CandidatePlayer
from airsenal.framework.utils import (
    session,
    NEXT_GAMEWEEK,
    CURRENT_SEASON,
    get_predicted_points,
    fastcopy,
    get_squad_value,
)
from copy import deepcopy

positions = ["FWD", "MID", "DEF", "GK"]  # front-to-back


def check_tag_valid(pred_tag, gameweek_range, season=CURRENT_SEASON, dbsession=session):
    """Check a prediction tag contains predictions for all the specified gameweeks."""
    # get unique gameweek and season values associated with pred_tag
    fixtures = (
        (
            dbsession.query(Fixture.season, Fixture.gameweek)
            .filter(PlayerPrediction.tag == pred_tag)
            .join(PlayerPrediction)
        )
        .distinct()
        .all()
    )
    pred_seasons = [f[0] for f in fixtures]
    pred_gws = [f[1] for f in fixtures]

    season_ok = all([s == season for s in pred_seasons])
    gws_ok = all([gw in pred_gws for gw in gameweek_range])

    return season_ok and gws_ok


def calc_points_hit(num_transfers, free_transfers):
    """
    Current rules say we lose 4 points for every transfer beyond
    the number of free transfers we have.
    Num transfers can be an integer, or "W", "F", "Bx", or "Tx"
    (wildcard, free hit, bench-boost or triple-caption).
    For Bx and Tx the "x" corresponds to the number of transfers
    in addition to the chip being played.
    """
    if num_transfers == "W" or num_transfers == "F":
        return 0
    elif isinstance(num_transfers, int):
        return max(0, 4 * (num_transfers - free_transfers))
    elif (num_transfers.startswith("B") or num_transfers.startswith("T")) and len(
        num_transfers
    ) == 2:
        num_transfers = int(num_transfers[-1])
        return max(0, 4 * (num_transfers - free_transfers))
    else:
        raise RuntimeError(
            "Unexpected argument for num_transfers {}".format(num_transfers)
        )


def calc_free_transfers(num_transfers, prev_free_transfers):
    """
    We get one extra free transfer per week, unless we use a wildcard or
    free hit, but we can't have more than 2.  So we should only be able
    to return 1 or 2.
    """
    if num_transfers == "W" or num_transfers == "F":
        return 1
    elif isinstance(num_transfers, int):
        return max(1, min(2, 1 + prev_free_transfers - num_transfers))
    elif (num_transfers.startswith("B") or num_transfers.startswith("T")) and len(
        num_transfers
    ) == 2:
        # take the 'x' out of Bx or Tx
        num_transfers = int(num_transfers[-1])
        return max(1, min(2, 1 + prev_free_transfers - num_transfers))
    else:
        raise RuntimeError(
            "Unexpected argument for num_transfers {}".format(num_transfers)
        )


def get_starting_squad(fpl_team_id=None):
    """
    use the transactions table in the db
    """

    if not fpl_team_id:
        # use the most recent transaction in the table
        most_recent = (
            session.query(Transaction)
            .order_by(Transaction.id.desc())
            .filter_by(free_hit=0)
            .first()
        )
        fpl_team_id = most_recent.fpl_team_id
    print("Getting starting squad for {}".format(fpl_team_id))
    s = Squad()
    # Don't include free hit transfers as they only apply for the week the
    # chip is activated
    transactions = (
        session.query(Transaction)
        .order_by(Transaction.gameweek, Transaction.id)
        .filter_by(fpl_team_id=fpl_team_id)
        .filter_by(free_hit=0)
        .all()
    )
    for trans in transactions:
        if trans.bought_or_sold == -1:
            s.remove_player(trans.player_id, price=trans.price)
        else:
            # within an individual transfer we can violate the budget and squad
            # constraints, as long as the final squad for that gameweek obeys them
            s.add_player(
                trans.player_id,
                price=trans.price,
                season=trans.season,
                gameweek=trans.gameweek,
                check_budget=False,
                check_team=False,
            )
    return s


def get_discount_factor(next_gw, pred_gw, discount_type="exp", discount=14 / 15):
    """
    given the next gw and a predicted gw, retrieve discount factor. Either:
        - exp: discount**n_ahead (discount reduces each gameweek)
        - const: 1-(1-discount)*n_ahead (constant discount each gameweek, goes to
          zero at gw 15 with default discount)
    """
    allowed_types = ["exp", "const", "constant"]
    if discount_type not in allowed_types:
        raise Exception("unrecognised discount type, should be exp or const")

    if not next_gw:
        # during tests 'none' is passed as the root gw, default to zero so the
        # optimisation is done solely on pred_gw ahead.
        next_gw = pred_gw
    n_ahead = pred_gw - next_gw

    if discount_type in ["exp"]:
        score = discount ** n_ahead
    elif discount_type in ["const", "constant"]:
        score = max(1 - (1 - discount) * n_ahead, 0)

    return score


def get_baseline_prediction(gw_ahead, tag, fpl_team_id=None):
    """
    use current squad, and count potential score
    also return a cumulative total per gw, so we can abort if it
    looks like we're doing too badly.
    """
    squad = get_starting_squad(fpl_team_id=fpl_team_id)
    total = 0.0
    cum_total_per_gw = {}
    next_gw = NEXT_GAMEWEEK
    gameweeks = list(range(next_gw, next_gw + gw_ahead))
    for gw in gameweeks:
        score = squad.get_expected_points(gw, tag) * get_discount_factor(next_gw, gw)
        cum_total_per_gw[gw] = total + score
        total += score
    return total, cum_total_per_gw


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
    best_pid_out, best_pid_in = 0, 0
    ordered_player_lists = {}
    if verbose:
        print("Creating ordered player lists")
    for pos in ["GK", "DEF", "MID", "FWD"]:
        ordered_player_lists[pos] = get_predicted_points(
            gameweek=gameweek_range, position=pos, tag=tag
        )
    for p_out in squad.players:
        if update_func_and_args:
            # call function to update progress bar.
            # this was passed as a tuple (func, increment, pid)
            update_func_and_args[0](update_func_and_args[1], update_func_and_args[2])

        new_squad = fastcopy(squad)
        position = p_out.position
        if verbose:
            print("Removing player {}".format(p_out.player_id))
        new_squad.remove_player(p_out.player_id, season=season, gameweek=transfer_gw)
        for p_in in ordered_player_lists[position]:
            if p_in[0].player_id == p_out.player_id:
                continue  # no point in adding the same player back in
            added_ok = new_squad.add_player(
                p_in[0], season=season, gameweek=transfer_gw
            )
            if added_ok:
                if verbose:
                    print("Added player {}".format(p_in[0].name))
                break
            else:
                if verbose:
                    print("Failed to add {}".format(p_in[0].name))
        total_points = 0.0
        for gw in gameweek_range:
            if gw == bench_boost_gw:
                total_points += new_squad.get_expected_points(
                    gw, tag, bench_boost=True
                ) * get_discount_factor(root_gw, gw)
            elif gw == triple_captain_gw:
                total_points += new_squad.get_expected_points(
                    gw, tag, triple_captain=True
                ) * get_discount_factor(root_gw, gw)
            else:
                total_points += new_squad.get_expected_points(
                    gw, tag
                ) * get_discount_factor(root_gw, gw)
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
    root_gw=None,
    season=CURRENT_SEASON,
    update_func_and_args=None,
    bench_boost_gw=None,
    triple_captain_gw=None,
    verbose=False,
):
    """
    If we want to just make two transfers, it's not unfeasible to try all
    possibilities in turn.
    We will order the list of potential subs via the sum of expected points
    over a specified range of gameweeks.
    """
    if not gameweek_range:
        gameweek_range = [NEXT_GAMEWEEK]
        root_gw = NEXT_GAMEWEEK

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
                # call function to update progress bar.
                # this was passed as a tuple (func, increment, pid)
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
            # what positions do we need to fill?
            positions_needed = [pout_1.position, pout_2.position]

            # now loop over lists of players and add players back in
            for pin_1 in ordered_player_lists[positions_needed[0]]:
                if (
                    pin_1[0].player_id == pout_1.player_id
                    or pin_1[0].player_id == pout_2.player_id
                ):
                    continue  # no point in adding same player back in
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
                        continue  # no point in adding same player back in
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
                                ) * get_discount_factor(root_gw, gw)
                            elif gw == triple_captain_gw:
                                total_points += new_squad_add_2.get_expected_points(
                                    gw, tag, triple_captain=True
                                ) * get_discount_factor(root_gw, gw)
                            else:
                                total_points += new_squad_add_2.get_expected_points(
                                    gw, tag
                                ) * get_discount_factor(root_gw, gw)
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
    root_gw=None,
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
            new_squad.remove_player(
                removed_players[-1], season=season, gameweek=transfer_gw
            )
        predicted_points = {}
        for pos in set(positions_needed):
            predicted_points[pos] = get_predicted_points(
                position=pos, gameweek=gw_range, tag=tag
            )
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

        # calculate the score
        total_points = 0.0
        for gw in gw_range:
            if gw == bench_boost_gw:
                total_points += new_squad.get_expected_points(
                    gw, tag, bench_boost=True
                ) * get_discount_factor(root_gw, gw)
            elif gw == triple_captain_gw:
                total_points += new_squad.get_expected_points(
                    gw, tag, triple_captain=True
                ) * get_discount_factor(root_gw, gw)
            else:
                total_points += new_squad.get_expected_points(
                    gw, tag
                ) * get_discount_factor(root_gw, gw)
        if total_points > best_score:
            best_score = total_points
            best_pid_out = removed_players
            best_pid_in = [ap.player_id for ap in added_players]
            best_squad = new_squad
        # end of loop over n_iter
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
):
    """
    Return a new squad and a dictionary {"in": [player_ids],
                                        "out":[player_ids]}
    """
    transfer_dict = {}
    # deal with triple_captain or free_hit
    triple_captain_gw = None
    bench_boost_gw = None
    if isinstance(num_transfers, str) and num_transfers.startswith("T"):
        num_transfers = int(num_transfers[1])
        triple_captain_gw = gameweeks[0]
    elif isinstance(num_transfers, str) and num_transfers.startswith("B"):
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

    elif num_transfers == "W" or num_transfers == "F":
        players_out = [p.player_id for p in squad.players]
        budget = get_squad_value(squad)
        if num_transfers == "F":
            # for free hit, only use one week to optimize
            gameweeks = [gameweeks[0]]
        new_squad = make_new_squad(
            budget, num_iter, tag, gameweeks, update_func_and_args=update_func_and_args
        )
        players_in = [p.player_id for p in new_squad.players]
        transfer_dict = {"in": players_in, "out": players_out}

    else:
        raise RuntimeError(
            "Unrecognized value for num_transfers: {}".format(num_transfers)
        )

    # get the expected points total for next gameweek
    points = (
        new_squad.get_expected_points(
            gameweeks[0],
            tag,
            triple_captain=(triple_captain_gw is not None),
            bench_boost=(bench_boost_gw is not None),
        )
        * get_discount_factor(root_gw, gameweeks[0])
    )

    if num_transfers == "F":
        # Free Hit changes don't apply to next gameweek, so return the original squad
        return squad, transfer_dict, points
    else:
        return new_squad, transfer_dict, points


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
            # call function to update progress bar.
            # this was passed as a tuple (func, increment, pid)
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
                score += t.get_expected_points(
                    gw, tag, bench_boost=True
                ) * get_discount_factor(gw_range[0], gw)
            elif gw == triple_captain_gw:
                score += t.get_expected_points(
                    gw, tag, triple_captain=True
                ) * get_discount_factor(gw_range[0], gw)
            else:
                score += t.get_expected_points(gw, tag) * get_discount_factor(
                    gw_range[0], gw
                )
        if score > best_score:
            best_score = score
            best_squad = t

    if verbose:
        print("====================================\n")
        print(best_squad)
        print(best_score)
    return best_squad


def fill_suggestion_table(baseline_score, best_strat, season, fpl_team_id):
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
                ts.fpl_team_id = fpl_team_id
                ts.chip_played = best_strat["chips_played"][gameweek]
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


def get_num_increments(num_transfers, num_iterations=100):
    """
    how many steps for the progress bar for this strategy
    """
    if (
        isinstance(num_transfers, str)
        and (num_transfers.startswith("B") or num_transfers.startswith("T"))
        and len(num_transfers) == 2
    ):
        num_transfers = int(num_transfers[1])

    if (
        num_transfers == "W"
        or num_transfers == "F"
        or (isinstance(num_transfers, int) and num_transfers > 2)
    ):
        # wildcard or free hit or >2 - needs num_iterations iterations
        return num_iterations

    elif num_transfers == 0:
        return 1

    elif num_transfers == 1:
        # single transfer - 15 increments (replace each player in turn)
        return 15
    elif num_transfers == 2:
        # remove each pair of players - 15*7=105 combinations
        return 105
    else:
        print("Unrecognized num_transfers: {}".format(num_transfers))
        return 1


def next_week_transfers(
    strat,
    max_total_hit=None,
    allow_unused_transfers=True,
    max_transfers=2,
    chips={"chips_allowed": [], "chip_to_play": None},
):
    """Given a previous strategy and some optimisation constraints, determine the valid
    options for the number of transfers (or chip played) in the following gameweek.

    strat is a tuple (free_transfers, hit_so_far, strat_dict)
    strat_dict must have key chips_played, which is a dict indexed by gameweek with
    possible values None, "wildcard", "free_hit", "bench_boost" or triple_captain"
    """
    # check that the 'chips' dict we are given makes sense:
    if (
        "chips_allowed" in chips.keys()
        and len(chips["chips_allowed"]) > 0
        and "chip_to_play" in chips.keys()
        and chips["chip_to_play"]
    ):
        raise RuntimeError(
            "Cannot allow {} in the same week as we play {}".format(
                chips["chips_allowed"], chips["chip_to_play"]
            )
        )
    ft_available, hit_so_far, strat_dict = strat
    chip_history = strat_dict["chips_played"]

    if not allow_unused_transfers and ft_available == 2:
        # Force at least 1 free transfer.
        # NOTE: This will exclude the baseline strategy when allow_unused_transfers
        # is False. Re-add it outside this function in that case.
        ft_choices = list(range(1, max_transfers + 1))
    else:
        ft_choices = list(range(max_transfers + 1))

    if max_total_hit is not None:
        ft_choices = [
            nt
            for nt in ft_choices
            if hit_so_far + calc_points_hit(nt, ft_available) <= max_total_hit
        ]

    allow_wildcard = (
        "chips_allowed" in chips.keys()
        and "wildcard" in chips["chips_allowed"]
        and "wildcard" not in chip_history.values()
    )
    allow_free_hit = (
        "chips_allowed" in chips.keys()
        and "free_hit" in chips["chips_allowed"]
        and "free_hit" not in chip_history.values()
    )
    allow_bench_boost = (
        "chips_allowed" in chips.keys()
        and "bench_boost" in chips["chips_allowed"]
        and "bench_boost" not in chip_history.values()
    )
    allow_triple_captain = (
        "chips_allowed" in chips.keys()
        and "triple_captain" in chips["chips_allowed"]
        and "triple_captain" not in chip_history.values()
    )

    # if we are definitely going to play a wildcard or free_hit deal with
    # that first
    if "chip_to_play" in chips.keys() and chips["chip_to_play"] == "wildcard":
        new_transfers = ["W"]
    elif "chip_to_play" in chips.keys() and chips["chip_to_play"] == "free_hit":
        new_transfers = ["F"]
    # for triple captain or bench boost, we can still do ft_choices transfers
    elif "chip_to_play" in chips.keys() and chips["chip_to_play"] == "triple_captain":
        new_transfers = [f"T{nt}" for nt in ft_choices]
    elif "chip_to_play" in chips.keys() and chips["chip_to_play"] == "bench_boost":
        new_transfers = [f"B{nt}" for nt in ft_choices]
    else:
        # no chip definitely played, but some might be allowed
        new_transfers = [nt for nt in ft_choices]  # make a copy
        if allow_wildcard:
            new_transfers.append("W")
        if allow_free_hit:
            new_transfers.append("F")
        if allow_bench_boost:
            new_transfers += [f"B{nt}" for nt in ft_choices]
        if allow_triple_captain:
            new_transfers += [f"T{nt}" for nt in ft_choices]

    new_points_hits = [
        hit_so_far + calc_points_hit(nt, ft_available) for nt in new_transfers
    ]
    new_ft_available = [calc_free_transfers(nt, ft_available) for nt in new_transfers]

    # return list of (num_transfers, free_transfers, hit_so_far) tuples for each new
    # strategy
    return list(zip(new_transfers, new_ft_available, new_points_hits))


def count_expected_outputs(
    gw_ahead,
    next_gw=NEXT_GAMEWEEK,
    free_transfers=1,
    max_total_hit=None,
    allow_unused_transfers=True,
    max_transfers=2,
    chip_gw_dict={},
):
    """
    Count the number of possible transfer and chip strategies for gw_ahead gameweeks
    ahead, subject to:
    * Start with free_transfers free transfers.
    * Spend a max of max_total_hit points on transfers across whole period
    (None for no limit)
    * Allow playing the chips which have their allow_xxx argument set True
    * Exclude strategies that waste free transfers (make 0 transfers if 2 free tramsfers
    are available), if allow_unused_transfers is False.
    * Make a maximum of max_transfers transfers each gameweek.
    * Each chip only allowed once.
    """

    init_strat_dict = {
        "players_in": {},
        "chips_played": {},
    }
    init_free_transfers = free_transfers  # used below for baseline strategy logic
    strategies = [(init_free_transfers, 0, init_strat_dict)]

    for gw in range(next_gw, next_gw + gw_ahead):
        new_strategies = []
        for s in strategies:
            free_transfers = s[0]
            chips_for_gw = chip_gw_dict[gw] if gw in chip_gw_dict.keys() else {}
            possibilities = next_week_transfers(
                s,
                max_total_hit=max_total_hit,
                max_transfers=max_transfers,
                allow_unused_transfers=allow_unused_transfers,
                chips=chips_for_gw,
            )

            for n_transfers, new_free_transfers, new_hit in possibilities:
                # make a copy of the strategy up to this point, then add on this gw
                new_dict = deepcopy(s[2])

                # update dummy strat dict
                if n_transfers == "W":
                    # add dummy values to transfer dict for 15 possible transfers
                    new_dict["players_in"][gw] = [1] * 15
                    new_dict["chips_played"][gw] = "wildcard"
                elif n_transfers == "F":
                    # add dummy values to transfer dict for 15 possible transfers
                    new_dict["players_in"][gw] = [1] * 15
                    new_dict["chips_played"][gw] = "free_hit"
                else:
                    if isinstance(n_transfers, str) and (
                        n_transfers.startswith("T") or n_transfers.startswith("B")
                    ):
                        if n_transfers[0] == "T":
                            new_dict["chips_played"][gw] = "triple_captain"
                        elif n_transfers[0] == "B":
                            new_dict["chips_played"][gw] = "bench_boost"
                        n_transfers = int(n_transfers[1])
                    # add dummy values to transfer dict for n_transfers transfers
                    new_dict["players_in"][gw] = [1] * n_transfers

                new_strategies.append((new_free_transfers, new_hit, new_dict))

        strategies = new_strategies

    # if allow_unused_transfers is False baseline of no transfers will be removed above,
    # add it back in here, apart from edge cases where it's already included.
    if not allow_unused_transfers and (
        gw_ahead > 1 or (gw_ahead == 1 and init_free_transfers == 2)
    ):
        baseline_strat_dict = {
            "players_in": {gw: [] for gw in range(next_gw, next_gw + gw_ahead)},
            "chips_played": {},
        }
        baseline_dict = (2, 0, baseline_strat_dict)
        strategies.insert(0, baseline_dict)
    return len(strategies)
