#!/usr/bin/env python

import argparse
import sys

from airsenal.framework.optimization_squad import make_new_squad
from airsenal.framework.optimization_utils import (
    DEFAULT_SUB_WEIGHTS,
    check_tag_valid,
    fill_initial_suggestion_table,
    fill_initial_transaction_table,
    get_discounted_squad_score,
)
from airsenal.framework.season import CURRENT_SEASON
from airsenal.framework.squad import Squad
from airsenal.framework.utils import (
    NEXT_GAMEWEEK,
    fetcher,
    get_latest_prediction_tag,
    get_max_gameweek,
)

positions = ["FWD", "MID", "DEF", "GK"]  # front-to-back


def fill_initial_squad(
    tag: str,
    gw_range: list[int],
    season: str,
    fpl_team_id: int,
    budget: int = 1000,
    remove_zero: bool = True,
    sub_weights: dict = DEFAULT_SUB_WEIGHTS,
    num_generations: int = 100,
    population_size: int = 100,
    crossover_prob: float = 0.7,
    mutation_prob: float = 0.3,
    crossover_indpb: float = 0.5,
    mutation_indpb: float = 0.1,
    tournament_size: int = 3,
    verbose: bool = True,
    is_replay: bool = False,  # for replaying seasons
) -> Squad:
    best_squad = make_new_squad(
        gw_range,
        tag,
        budget=budget,
        season=season,
        remove_zero=remove_zero,
        sub_weights=sub_weights,
        population_size=population_size,
        generations=num_generations,
        crossover_prob=crossover_prob,
        mutation_prob=mutation_prob,
        crossover_indpb=crossover_indpb,
        mutation_indpb=mutation_indpb,
        tournament_size=tournament_size,
        verbose=verbose,
    )

    if best_squad is None:
        msg = (
            "best_squad is None: make_new_squad failed to generate a valid team or "
            "something went wrong with the squad expected points calculation."
        )
        raise RuntimeError(msg)

    gw_start = gw_range[0]
    optimised_score = get_discounted_squad_score(
        best_squad,
        gw_range,
        tag,
        gw_start,
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
    # parameters for deap optimization
    parser.add_argument(
        "--num_generations",
        help="number of generations",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--population_size",
        help="number of candidate solutions per generation",
        type=int,
        default=100,
    )
    # parameters for "deap" optimization
    parser.add_argument(
        "--crossover_prob",
        help="crossover probability",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--mutation_prob",
        help="mutation probability",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "--crossover_indpb",
        help="independent probability for each attribute to be exchanged in crossover",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--mutation_indpb",
        help="independent probability for each attribute to be mutated",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--tournament_size",
        help="size of tournament for tournament selection",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--no_subs",
        help="Don't include points contribution from substitutes",
        action="store_true",
    )
    parser.add_argument(
        "--include_zero",
        help="Include players with zero predicted points",
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
    num_generations = args.num_generations
    population_size = args.population_size
    crossover_prob = args.crossover_prob
    mutation_prob = args.mutation_prob
    crossover_indpb = args.crossover_indpb
    mutation_indpb = args.mutation_indpb
    tournament_size = args.tournament_size
    remove_zero = not args.include_zero
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
        remove_zero=remove_zero,
        sub_weights=sub_weights,
        num_generations=num_generations,
        population_size=population_size,
        crossover_prob=crossover_prob,
        mutation_prob=mutation_prob,
        crossover_indpb=crossover_indpb,
        mutation_indpb=mutation_indpb,
        tournament_size=tournament_size,
        verbose=True,
        is_replay=args.is_replay,
    )
