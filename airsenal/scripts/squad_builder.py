#!/usr/bin/env python

import os
import sys

import random
import argparse

from ..framework.utils import *
from ..framework.squad import Squad, TOTAL_PER_POSITION
from ..framework.player import CandidatePlayer
from ..framework.optimization_utils import make_new_squad

positions = ["FWD", "MID", "DEF", "GK"]  # front-to-back


def main():
    parser = argparse.ArgumentParser(description="make a squad from scratch")
    parser.add_argument(
        "--num_iterations", help="number of iterations", type=int, default=10
    )
    parser.add_argument(
        "--budget", help="budget, in 0.1 millions", type=int, default=1000
    )
    parser.add_argument("--season", help="season, in format e.g. 1819")
    parser.add_argument("--gw_start", help="gameweek to start from", type=int)
    parser.add_argument(
        "--num_gw", help="how many gameweeks to consider", type=int, default=3
    )
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
        gw_start = NEXT_GAMEWEEK
    ## get predicted points
    gw_range = list(range(gw_start, min(38, gw_start + args.num_gw)))
    tag = get_latest_prediction_tag(season)
    best_squad = make_new_squad(args.budget, num_iterations, tag, gw_range, season)
    points = best_squad.get_expected_points(gw_start, tag)
    print("---------------------")
    print("Best expected points for gameweek {}: {}".format(gw_start, points))
    print("---------------------")
    print(best_squad)
