#!/usr/bin/env python

"""
usage:
python team_optimizer.py <num_weeks_ahead> <num_iterations>
output for each strategy tried is going to be a dict
{ "total_points": <float>,
"points_per_gw": {<gw>: <float>, ...},
"players_sold" : {<gw>: [], ...},
"players_bought" : {<gw>: [], ...}
}
"""


import os
import sys

sys.path.append("..")
import random

from framework.utils import *
from framework.team import Team, TOTAL_PER_POSITION
from framework.player import CandidatePlayer

positions = ["FWD", "MID", "DEF", "GK"]  # front-to-back


def get_starting_team():
    """
    use the transactions table in the db
    """
    team_ids = get_current_players()
    t = Team()
    for pid in team_ids:
        t.add_player(pid)
    return t


def get_baseline_prediction(team, gw_ahead, method="AIv1"):
    """
    use current team, and count potential score
    also return a cumulative total per gw, so we can abort if it
    looks like we're doing too badly.
    """

    total = 0.0
    cum_total_per_gw = {}
    next_gw = get_next_gameweek()
    gameweeks = list(range(next_gw, next_gw + gw_ahead))
    for gw in gameweeks:
        score = team.get_expected_points(gw, method)
        cum_total_per_gw[gw] = total + score
        total += score
    return total, cum_total_per_gw


def make_optimimum_substitution(team, method="AIv1", gameweek_range=None):
    """
    If we want to just make one sub, it's not unfeasible to try all
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
            gameweek=gameweek_range, position=pos, method=method
        )
    for p_out in team.players:
        new_team = copy.deepcopy(team)
        position = p_out.position
        new_team.remove_player(p_out.player_id)
        for p_in in ordered_player_lists[position]:
            if p_in[0] == p_out.player_id:
                continue  # no point in adding the same player back in
            added_ok = new_team.add_player(p_in[0])
            if added_ok:
                break
        total_points = 0.
        for gw in gameweek_range:
            total_points += new_team.get_expected_points(gw, method)
        if total_points > best_score:
            best_score = total_points
            best_pid_out = p_out.player_id
            best_pid_in = p_in[0]
            best_team = new_team
    return best_team, best_pid_out, best_pid_in


def make_random_substitutions(team, nsubs=1, method="AIv1", gameweek=None):
    """
    choose a random player to sub out, and then get the player with the best
    expected score for the next gameweek that we can to fill their place.
    """
    new_team = copy.deepcopy(team)
    if not gameweek:
        gameweek = get_next_gameweek()
    players_to_remove = []  # this is the index within the team
    removed_players = []  # this is the player_ids
    ## order the players in the team by predicted_points - least-to-most
    player_list = []
    for p in team.players:
        p.calc_predicted_points(method)
        player_list.append((p.player_id, p.predicted_points[method][gameweek]))
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
            position=pos, gameweek=gameweek, method=method
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
                removed_ok = new_team.remove_player(ap)
            added_players = []
    return new_team, removed_players, added_players


def apply_strategy(strat, starting_team, method="AIv1", baseline_dict=None):
    """
    apply a set of substitutions over a number of gameweeks, and
    total up the score, taking into account points hits.
    """
    print("Trying strategy {}".format(strat))
    new_team = copy.deepcopy(starting_team)
    strategy_output = {
        "total_score": -1 * strat[1],  # points hit from this strategy
        "points_per_gw": {},
        "players_in": {},
        "players_out": {},
    }
    for gw in sorted(strat[0].keys()):  # make sure we go through gameweeks in order
        if strat[0][gw] == 0:  # no transfers that gameweek
            score = new_team.get_expected_points(gw)
            rp, ap = [], []
        else:
            new_team, rp, ap = make_random_substitutions(
                new_team, strat[0][gw], method, gw
            )
            score = new_team.get_expected_points(gw)
        ## if we're ever >5 points below the baseline, bail out!
        strategy_output["total_score"] += score
        if baseline_dict and baseline_dict[gw] - strategy_output["total_score"] > 5:
            return strategy_output
        strategy_output["points_per_gw"][gw] = score
        strategy_output["players_in"][gw] = ap
        strategy_output["players_out"][gw] = rp
    print("Total score: {}".format(strategy_output["total_score"]))
    return strategy_output


if __name__ == "__main__":

    num_weeks_ahead = int(sys.argv[-2])
    num_iterations = int(sys.argv[-1])

    ## get our current team
    t = get_starting_team()
    ## if we do nothing...
    baseline, baseline_dict = get_baseline_prediction(t, num_weeks_ahead)
    print("Baseline score for next {} gw:  {}".format(num_weeks_ahead, baseline))
    strategies = generate_transfer_strategies(num_weeks_ahead)
    best_score = baseline
    best_strat = None
    for iteration in range(num_iterations):
        t = get_starting_team()
        for s in strategies:
            strategy_output = apply_strategy(
                strat=s, starting_team=t, method="AIv1", baseline_dict=baseline_dict
            )
            if strategy_output["total_score"] > best_score:
                best_score = strategy_output["total_score"]
                best_strat = strategy_output

    print("====================================\n")
    print("Baseline score: {}".format(baseline))
    print(" best strategy")
    print(best_strat)
