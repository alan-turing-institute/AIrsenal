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
    # General parameters
    parser.add_argument(
        "--budget", help="budget, in 0.1 millions", type=int, default=1000
    )
    parser.add_argument("--season", help="season, in format e.g. 1819")
    parser.add_argument("--gw_start", help="gameweek to start from", type=int)
    parser.add_argument(
        "--num_gw", help="how many gameweeks to consider", type=int, default=3
    )
    parser.add_argument(
        "--algorithm",
        help="Which optimization algorithm to use - 'normal' or 'genetic'",
        type=str,
        default="normal",
    )
    # parameters for "normal" optimization
    parser.add_argument(
        "--num_iterations",
        help="number of iterations (normal algorithm only)",
        type=int,
        default=10,
    )
    # parameters for "pygmo" optimization
    parser.add_argument(
        "--num_generations",
        help="number of generations (genetic only)",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--population_size",
        help="number of candidate solutions per generation (genetic only)",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--no_subs",
        help="Don't include points contribution from substitutes (genetic only)",
        action="store_true",
    )
    parser.add_argument(
        "--include_zero",
        help="Include players with zero predicted points (genetic only)",
        action="store_true",
    )
    parser.add_argument(
        "--gw_weight_type",
        help="'constant' to treat all gameweeks equally, or 'linear' to reduce weight of gameweeks with time (genetic only)",
        type=str,
        default="linear",
    )

    args = parser.parse_args()
    if args.season:
        season = args.season
    else:
        season = get_current_season()
    budget = args.budget
    if args.gw_start:
        gw_start = args.gw_start
    else:
        gw_start = NEXT_GAMEWEEK

    gw_range = list(range(gw_start, min(38, gw_start + args.num_gw)))
    tag = get_latest_prediction_tag(season)

    if args.algorithm == "normal":
        from ..framework.optimization_utils import make_new_team

        num_iterations = args.num_iterations
        best_team = make_new_team(args.budget, num_iterations, tag, gw_range, season)

    elif args.algorithm == "genetic":
        import pygmo as pg
        from ..framework.optimization_pygmo import make_new_team

        num_generations = args.num_generations
        population_size = args.population_size
        remove_zero = not args.include_zero
        gw_weight_type = args.gw_weight_type
        uda = pg.sga(gen=num_generations)
        if args.no_subs:
            sub_weights = {"GK": 0, "Outfield": (0, 0, 0)}
        else:
            sub_weights = {"GK": 0.01, "Outfield": (0.4, 0.1, 0.02)}

        best_team = make_new_team(
            gw_range,
            tag,
            budget=budget,
            season=season,
            remove_zero=remove_zero,
            sub_weights=sub_weights,
            uda=uda,
            population_size=population_size,
            gw_weight_type=gw_weight_type,
        )
    else:
        raise ValueError("'algorithm' must be 'normal' or 'genetic'")

    points = best_team.get_expected_points(gw_start, tag)
    print("---------------------")
    print("Best expected points for gameweek {}: {}".format(gw_start, points))
    print("---------------------")
    print(best_team)
