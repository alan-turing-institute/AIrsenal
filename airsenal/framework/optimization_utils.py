"""
functions to optimize the transfers for N weeks ahead
"""

import os
import sys


import random
from datetime import datetime

from .utils import *
from .schema import TransferSuggestion, PlayerPrediction
from .team import Team, TOTAL_PER_POSITION
from .player import CandidatePlayer

positions = ["FWD", "MID", "DEF", "GK"]  # front-to-back


def generate_transfer_strategies(gw_ahead, transfers_last_gw=1, max_total_hit=None):
    """
    Constraint: we want to take no more than a 4-point hit each week.
    So, for each gameweek, we can make 0, 1, or 2 changes, or, if we made 0
    the previous week, we can make 3.
    Generate all possible sequences, for N gameweeks ahead, and return along
    with the total points hit.
    i.e. return value is a list of tuples:
        [({gw:ntransfer, ...},points_hit), ... ]
    """
    next_gw = get_next_gameweek()
    strategy_list = []
    possibilities = list(range(4)) if transfers_last_gw == 0 else list(range(3))
    strategies = [
        ({next_gw: i}, 4 * (max(0, i - (1 + int(transfers_last_gw == 0)))))
        for i in possibilities
    ]
    for gw in range(next_gw + 1, next_gw + gw_ahead):
        new_strategies = []
        for s in strategies:
            possibilities = list(range(4)) if s[0][gw - 1] == 0 else list(range(3))
            hit_so_far = s[1]
            for p in possibilities:
                new_dict = {}
                for k, v in s[0].items():
                    new_dict[k] = v
                new_dict[gw] = p
                new_hit = hit_so_far + 4 * (max(0, p - (1 + int(s[0][gw - 1] == 0))))
                new_strategies.append((new_dict, new_hit))
        strategies = new_strategies
    if max_total_hit:
        strategies = [s for s in strategies if s[1] <= max_total_hit]
    return strategies


def get_starting_team():
    """
    use the transactions table in the db
    """
    team_ids = get_current_players()
    t = Team()
    for pid in team_ids:
        t.add_player(pid)
    return t


def get_baseline_prediction(gw_ahead, tag):
    """
    use current team, and count potential score
    also return a cumulative total per gw, so we can abort if it
    looks like we're doing too badly.
    """
    team = get_starting_team()
    total = 0.0
    cum_total_per_gw = {}
    next_gw = get_next_gameweek()
    gameweeks = list(range(next_gw, next_gw + gw_ahead))
    for gw in gameweeks:
        score = team.get_expected_points(gw, tag)
        cum_total_per_gw[gw] = total + score
        total += score
    return total, cum_total_per_gw


def make_optimum_transfer(team, tag, gameweek_range=None, season="1819"):
    """
    If we want to just make one transfer, it's not unfeasible to try all
    possibilities in turn.


    We will order the list of potential transfers via the sum of
    expected points over a specified range of gameweeks.
    """
    if not gameweek_range:
        gameweek_range = [get_next_gameweek()]
    best_score = 0.
    best_pid_out, best_pid_in = 0, 0
    ordered_player_lists = {}
    for pos in ["GK", "DEF", "MID", "FWD"]:
        ordered_player_lists[pos] = get_predicted_points(
            gameweek=gameweek_range, position=pos, tag=tag
        )
    for p_out in team.players:
        new_team = copy.deepcopy(team)
        position = p_out.position
#        print("Removing player {}".format(p_out.name))
        new_team.remove_player(p_out.player_id)
        for p_in in ordered_player_lists[position]:
            if p_in[0].player_id == p_out.player_id:
                continue  # no point in adding the same player back in
#            print("Trying to add player {}".format(p_in[0]))
            added_ok = new_team.add_player(p_in[0])
            if added_ok:
                break
        total_points = 0.
        for gw in gameweek_range:
            total_points += new_team.get_expected_points(gw, tag)
        if total_points > best_score:
            best_score = total_points
            best_pid_out = p_out.player_id
            best_pid_in = p_in[0].player_id
            best_team = new_team
    return best_team, [best_pid_out], [best_pid_in]


def make_optimum_double_transfer(team, tag, gameweek_range=None, season="1819"):
    """
    If we want to just make two transfers, it's not unfeasible to try all
    possibilities in turn.
    We will order the list of potential subs via the sum of expected points
    over a specified range of gameweeks.
    """
    if not gameweek_range:
        gameweek_range = [get_next_gameweek()]
    best_score = 0.
    best_pid_out, best_pid_in = 0, 0
    ordered_player_lists = {}
    for pos in ["GK", "DEF", "MID", "FWD"]:
        ordered_player_lists[pos] = get_predicted_points(
            gameweek=gameweek_range, position=pos, tag=tag
        )

    for i in range(len(team.players)-1):
        positions_needed = []
        pout_1 = team.players[i]

        new_team_remove_1 = copy.deepcopy(team)
        new_team_remove_1.remove_player(pout_1.player_id)
#        print("Removing player 1 {}".format(pout_1.name))
        for j in range(i+1, len(team.players)):
            pout_2 = team.players[j]
            new_team_remove_2 = copy.deepcopy(new_team_remove_1)
            new_team_remove_2.remove_player(pout_2.player_id)
            print("Removing players {} {}".format(i,j))
#            print("Removing player 2 {}".format(pout_2.name))
#            print("==> number of players {}".format(len(new_team_remove_2.players)))
            ## what positions do we need to fill?
            positions_needed = [pout_1.position, pout_2.position]

            # now loop over lists of players and add players back in
            for pin_1 in ordered_player_lists[positions_needed[0]]:
                if pin_1[0].player_id == pout_1.player_id \
                   or pin_1[0].player_id == pout_2.player_id:
                    continue   ## no point in adding same player back in
                new_team_add_1 = copy.deepcopy(new_team_remove_2)
                added_1_ok = new_team_add_1.add_player(pin_1[0])
                if not added_1_ok:
                    continue
#                print("==> number of players ADD1{}".format(len(new_team_add_1.players)))
                for pin_2 in ordered_player_lists[positions_needed[1]]:
                    new_team_add_2 = copy.deepcopy(new_team_add_1)
                    if pin_2[0] == pin_1[0] or \
                       pin_2[0].player_id == pout_1.player_id or \
                       pin_2[0].player_id == pout_2.player_id:
                        continue ## no point in adding same player back in
                    added_2_ok = new_team_add_2.add_player(pin_2[0])
                    if added_2_ok:
                        # calculate the score
                        total_points = 0.
                        for gw in gameweek_range:
                            total_points += new_team_add_2.get_expected_points(gw, tag)
                        if total_points > best_score:
                            best_score = total_points
                            best_pid_out = [pout_1.player_id,pout_2.player_id]
                            best_pid_in = [pin_1[0].player_id, pin_2[0].player_id]
                            best_team = new_team_add_2
                        break

    return best_team, best_pid_out, best_pid_in


def make_random_transfers(team, tag, nsubs=1, gw_range=None):
    """
    choose a random player to sub out, and then get the player with the best
    expected score for the next gameweek that we can to fill their place.
    """
    new_team = copy.deepcopy(team)
    if not gw_range:
        gw_range = [get_next_gameweek()]
    players_to_remove = []  # this is the index within the team
    removed_players = []  # this is the player_ids
    ## order the players in the team by predicted_points - least-to-most
    player_list = []
    for p in team.players:
        p.calc_predicted_points(tag)
        player_list.append((p.player_id, p.predicted_points[tag][gw_range[0]]))
    player_list.sort(key=itemgetter(1), reverse=False)
    while len(players_to_remove) < nsubs:
        index = int(random.triangular(0, len(player_list), 0))
        #        index = random.randint(0,len(player_list),0)
        if not index in players_to_remove:
            players_to_remove.append(index)

    positions_needed = []
    for p in players_to_remove:
        positions_needed.append(team.players[p].position)
        removed_players.append(team.players[p].player_id)
        new_team.remove_player(removed_players[-1])
    budget = new_team.budget
    predicted_points = {}
    for pos in set(positions_needed):
        predicted_points[pos] = get_predicted_points(
            position=pos, gameweek=gw_range, tag=tag
        )
    complete_team = False
    added_players = []
    while not complete_team:
        ## sample with a triangular PDF - preferentially select players near
        ## the start
        added_players = []
        for pos in positions_needed:
            index = int(random.triangular(0, len(predicted_points[pos]), 0))
            #            index = random.randint(0,len(predicted_points[pos])-1)
            pid_to_add = predicted_points[pos][index][0]
            added_ok = new_team.add_player(pid_to_add)
            if added_ok:
                added_players.append(pid_to_add)
        complete_team = new_team.is_complete()
        if not complete_team:  # take those players out again.
            for ap in added_players:
                removed_ok = new_team.remove_player(ap.player_id)
                if not removed_ok:
                    print("Problem removing {}".format(ap.name))
            added_players = []
    return new_team, removed_players, [ap.player_id for ap in added_players]


def apply_strategy(strat, exhaustive_double_transfer,
                   tag, baseline_dict=None, num_iter=1):
    """
    apply a set of transfers over a number of gameweeks, and
    total up the score, taking into account points hits.
    strat is a tuple, with the first element being the
    dictionary {gw:ntransfers,...} and the second element being
    the total points hit.
    """
    sid = make_strategy_id(strat)
    starting_team = get_starting_team()
    print("Trying strategy {}".format(strat))
    best_score = 0
    best_strategy_output = {}
    gameweeks = sorted(strat[0].keys())  # go through gameweeks in order
    for i in range(num_iter):
        print(" --> doing strategy {} iteration {}".format(sid, i))
        strategy_output = {
            "total_score": -1 * strat[1],  # points hit from this strategy
            "points_per_gw": {},
            "players_in": {},
            "players_out": {},
        }
        new_team = copy.deepcopy(starting_team)

        for igw, gw in enumerate(gameweeks):
            gw_range = gameweeks[igw:]  # range of gameweeks to end of window
            if strat[0][gw] == 0:  # no transfers that gameweek
                rp, ap = [], []
            elif strat[0][gw] == 1:  # one transfer - choose optimum
                new_team, rp, ap = make_optimum_transfer(new_team, tag, gw_range)
            elif strat[0][gw] == 2 and exhaustive_double_transfer:  # two transfer - choose optimum
                new_team, rp, ap = make_optimum_double_transfer(new_team, tag, gw_range)
            else:  # choose randomly
                new_team, rp, ap = make_random_transfers(
                    new_team, tag, strat[0][gw], gw_range
                )
            score = new_team.get_expected_points(gw, tag)
            ## if we're ever >5 points below the baseline, bail out!
            strategy_output["total_score"] += score
            if baseline_dict and baseline_dict[gw] - strategy_output["total_score"] > 5:
                break
            strategy_output["points_per_gw"][gw] = score
            strategy_output["players_in"][gw] = ap
            strategy_output["players_out"][gw] = rp
        ## end of loop over gameweeks
        if strategy_output["total_score"] > best_score:
            best_score = strategy_output["total_score"]
            best_strategy_output = strategy_output
    print("Total score: {}".format(best_strategy_output["total_score"]))
    return best_strategy_output


def fill_suggestion_table(baseline_score, best_strat):
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
        if v >= N:
            return True
    return False


def make_strategy_id(strategy):
    """
    Return a string that will identify a strategy - just concatenate
    the numbers of transfers per gameweek.
    """
    strat_id = ""
    for v in strategy[0].values():
        strat_id += str(v)
    return strat_id
