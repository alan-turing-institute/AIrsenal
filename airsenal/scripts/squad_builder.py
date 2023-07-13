#!/usr/bin/env python

import argparse
import sys
from typing import List

from airsenal.framework.optimization_squad import Squad, make_new_squad
from airsenal.framework.optimization_utils import (
    DEFAULT_SUB_WEIGHTS,
    check_tag_valid,
    fill_initial_suggestion_table,
    fill_initial_transaction_table,
    get_discounted_squad_score,
)
from airsenal.framework.season import CURRENT_SEASON
from airsenal.framework.utils import (
    NEXT_GAMEWEEK,
    fetcher,
    get_latest_prediction_tag,
    get_max_gameweek,
)

positions = ["FWD", "MID", "DEF", "GK"]  # front-to-back


def fill_initial_squad(
    tag: str,
    gw_range: List[int],
    season: str,
    fpl_team_id: int,
    budget: int = 1000,
    algorithm: str = "genetic",
    remove_zero: bool = True,
    sub_weights: dict = DEFAULT_SUB_WEIGHTS,
    num_generations: int = 100,
    population_size: int = 100,
    num_iterations: int = 10,
    verbose: bool = True,
    is_replay: bool = False,  # for replaying seasons
) -> Squad:
    if algorithm == "genetic":
        try:
            import pygmo as pg

            uda = pg.sga(gen=num_generations)
        except ModuleNotFoundError:
            print("pygmo not available. Defaulting to algorithm=normal instead")
            algorithm = "normal"
            uda = None
    else:
        uda = None

    gw_start = gw_range[0]
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

    optimised_score = get_discounted_squad_score(
        best_squad,
        gw_range,
        tag,
        gw_range[0],
        sub_weights=sub_weights,
    )
    next_points = best_squad.get_expected_points(gw_start, tag)
    print("---------------------")
    print(
        "Optimised total score (gameweeks",
        f"{min(gw_range)} to {max(gw_range)}): {optimised_score:.2f}",
    )
    print(f"Expected points for gameweek {gw_start}: {next_points:.2f}")
    print("---------------------")
    print(best_squad)

    fill_initial_suggestion_table(
        best_squad,
        fpl_team_id,
        tag,
        season=season,
        gameweek=gw_start,
    )
    if is_replay:
        # if simulating a previous season also add suggestions to transaction table
        # to imitate applying transfers
        fill_initial_transaction_table(
            best_squad,
            fpl_team_id,
            tag,
            season=season,
            gameweek=gw_start,
        )
    return best_squad


def main():
    parser = argparse.ArgumentParser(description="Make a squad from scratch")
    # General parameters
    parser.add_argument(
        "--budget", help="budget, in 0.1 millions", type=int, default=1000
    )
    parser.add_argument("--season", help="season, in format e.g. 1819")
    parser.add_argument("--gameweek_start", help="gameweek to start from", type=int)
    parser.add_argument(
        "--num_gameweeks", help="how many gameweeks to consider", type=int, default=3
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
    parser.add_argument(
        "--is_replay",
        help="Add suggested squad to the database (for replaying seasons)",
        action="store_true",
    )
    args = parser.parse_args()
    season = args.season or CURRENT_SEASON
    budget = args.budget
    if args.gameweek_start:
        gameweek_start = args.gameweek_start
    elif season == CURRENT_SEASON:
        gameweek_start = NEXT_GAMEWEEK
    else:
        gameweek_start = 1
    gw_range = list(
        range(
            gameweek_start,
            min(get_max_gameweek(season) + 1, gameweek_start + args.num_gameweeks),
        )
    )
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
    fpl_team_id = args.fpl_team_id or fetcher.FPL_TEAM_ID
    if args.no_subs:
        sub_weights = {"GK": 0, "Outfield": (0, 0, 0)}
    else:
        sub_weights = {"GK": 0.01, "Outfield": (0.4, 0.1, 0.02)}

    fill_initial_squad(
        tag=tag,
        gw_range=gw_range,
        season=season,
        fpl_team_id=fpl_team_id,
        budget=budget,
        algorithm=algorithm,
        remove_zero=remove_zero,
        sub_weights=sub_weights,
        num_generations=num_generations,
        population_size=population_size,
        num_iterations=num_iterations,
        verbose=verbose,
        is_replay=args.is_replay,
    )
