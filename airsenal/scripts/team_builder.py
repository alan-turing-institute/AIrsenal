#!/usr/bin/env python

import os
import sys

import random
import argparse

from ..framework.utils import *
from ..framework.team import Team, TOTAL_PER_POSITION
from ..framework.player import CandidatePlayer


positions = ["FWD", "MID", "DEF", "GK"]  # front-to-back

def main():
    parser = argparse.ArgumentParser(description="make a team from scratch")
    parser.add_argument("--num_iterations",help="number of iterations",type=int,default=10)
    parser.add_argument("--budget",help="budget, in 0.1 millions",type=int,default=1000)
    parser.add_argument("--season",help="season, in format e.g. 1819")
    parser.add_argument("--gw_start", help="gameweek to start from",type=int)
    parser.add_argument("--num_gw", help="how many gameweeks to consider",type=int, default=5)
    args = parser.parse_args()
    num_iterations = args.num_iterations
    if args.season:
        season = args.season
    else:
        season = get_current_season()
    budget = args.budget
    if args.gw_start:
        gw_start = args.gw_start
    else:
        gw_start = get_next_gameweek(season)
    ## get predicted points
    gw_range = list(range(gw_start, min(38,gw_start+args.num_gw)))

    tag = get_latest_prediction_tag(season)
    best_score = 0.
    best_team = None

    for iteration in range(num_iterations):
        predicted_points = {}
        t = Team(budget)
        # first iteration - fill up from the front
        for pos in positions:
            predicted_points[pos] = get_predicted_points(gameweek=gw_range,
                                                         position=pos,
                                                         tag=tag,
                                                         season=season)
            for pp in predicted_points[pos]:
                t.add_player(pp[0])
                if t.num_position[pos] == TOTAL_PER_POSITION[pos]:
                    break

        # presumably we didn't get a complete team now
        excluded_player_ids = []
        while not t.is_complete():
            # randomly swap out a player and replace with a cheaper one in the
            # same position
            player_to_remove = t.players[random.randint(0, len(t.players) - 1)]
            remove_cost = player_to_remove.current_price
            remove_position = player_to_remove.position
            t.remove_player(player_to_remove.player_id)
            excluded_player_ids.append(player_to_remove.player_id)
            for pp in predicted_points[player_to_remove.position]:
                if (
                    not pp[0] in excluded_player_ids
                ) or random.random() < 0.3:  # some chance to put player back
                    cp = CandidatePlayer(pp[0])
                    if cp.current_price >= remove_cost:
                        continue
                    else:
                        t.add_player(pp[0])
            # now try again to fill up the rest of the team
            num_missing_per_position = {}

            for pos in positions:
                num_missing = TOTAL_PER_POSITION[pos] - t.num_position[pos]
                if num_missing == 0:
                    continue
                for pp in predicted_points[pos]:
                    if pp[0] in excluded_player_ids:
                        continue
                    t.add_player(pp[0])
                    if t.num_position[pos] == TOTAL_PER_POSITION[pos]:
                        break
        # we have a complete team
        score = 0.
        for gw in gw_range:
            score += t.get_expected_points(gw,tag)
        if score > best_score:
            best_score = score
            best_team = t
        print(t)
        print("Score {}".format(score))
    print("====================================\n")
    print(best_team)
    print(best_score)
