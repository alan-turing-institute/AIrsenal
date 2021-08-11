#!/usr/bin/env python

import argparse
import sys

from airsenal.framework.utils import (
    NEXT_GAMEWEEK,
    get_latest_prediction_tag,
    fetcher,
)
from airsenal.framework.optimization_utils import (
    check_tag_valid,
    fill_initial_suggestion_table,
)
from airsenal.framework.season import get_current_season
from airsenal.framework.optimization_squad import make_new_squad

positions = ["FWD", "MID", "DEF", "GK"]  # front-to-back


def main():
    parser = argparse.ArgumentParser(description="Make a squad from scratch")
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
        default="genetic",
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
        "--verbose",
        help="Print details on optimsation progress",
        action="store_true",
    )
    parser.add_argument(
        "--fpl_team_id",
        help="ID for your FPL team",
        type=int,
    )
    args = parser.parse_args()
    season = args.season or get_current_season()
    budget = args.budget
    gw_start = args.gw_start or NEXT_GAMEWEEK
    gw_range = list(range(gw_start, min(38, gw_start + args.num_gw)))
    tag = get_latest_prediction_tag(season)
    if not check_tag_valid(tag, gw_range, season=season):
        print(
            "ERROR: Database does not contain predictions",
            "for all the specified optimsation gameweeks.\n",
            "Please run 'airsenal_run_prediction' first with the",
            "same input gameweeks and season you specified here.",
        )
        sys.exit(1)
    algorithm = args.algorithm
    num_iterations = args.num_iterations
    num_generations = args.num_generations
    population_size = args.population_size
    remove_zero = not args.include_zero
    verbose = args.verbose
    if args.no_subs:
        sub_weights = {"GK": 0, "Outfield": (0, 0, 0)}
    else:
        sub_weights = {"GK": 0.01, "Outfield": (0.4, 0.1, 0.02)}
    if algorithm == "genetic":
        try:
            import pygmo as pg

            uda = pg.sga(gen=num_generations)
        except ModuleNotFoundError as e:
            print(e)
            print("Defaulting to algorithm=normal instead")
            algorithm = "normal"
            uda = None
    else:
        uda = None

    best_squad = make_new_squad(
        gw_range,
        tag,
        budget=budget,
        season=season,
        algorithm=algorithm,
        remove_zero=remove_zero,
        sub_weights=sub_weights,
        uda=uda,
        population_size=population_size,
        num_iterations=num_iterations,
        verbose=verbose,
    )
    if best_squad is None:
        raise RuntimeError(
            "best_squad is None: make_new_squad failed to generate a valid team or "
            "something went wrong with the squad expected points calculation."
        )

    points = best_squad.get_expected_points(gw_start, tag)
    print("---------------------")
    print("Best expected points for gameweek {}: {}".format(gw_start, points))
    print("---------------------")
    print(best_squad)

    fpl_team_id = args.fpl_team_id or fetcher.FPL_TEAM_ID
    fill_initial_suggestion_table(
        best_squad,
        fpl_team_id,
        tag,
        season=season,
        gameweek=gw_start,
    )
