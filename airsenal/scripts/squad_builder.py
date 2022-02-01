#!/usr/bin/env python

import sys

from airsenal.framework.optimization_squad import make_new_squad
from airsenal.framework.optimization_utils import (
    check_tag_valid,
    fill_initial_suggestion_table,
)
from airsenal.framework.season import get_current_season
from airsenal.framework.utils import NEXT_GAMEWEEK, fetcher, get_latest_prediction_tag

positions = ["FWD", "MID", "DEF", "GK"]  # front-to-back


def optimise_squad(
    season: int,
    gameweek_start: int,
    num_gameweeks: int,
    budget: int,
    algorithm: str,
    num_iterations: int,
    num_generations: int,
    population_size: int,
    no_subs: bool,
    include_zero: bool,
    verbose: bool,
    fpl_team_id: int,
):
    season = season or get_current_season()
    budget = budget
    gameweek_start = gameweek_start or NEXT_GAMEWEEK
    gameweek_range = list(
        range(gameweek_start, min(38, gameweek_start + num_gameweeks))
    )
    tag = get_latest_prediction_tag(season)
    if not check_tag_valid(tag, gameweek_range, season=season):
        print(
            "ERROR: Database does not contain predictions",
            "for all the specified optimsation gameweeks.\n",
            "Please run 'airsenal_run_prediction' first with the",
            "same input gameweeks and season you specified here.",
        )
        sys.exit(1)
    algorithm = algorithm
    num_iterations = num_iterations
    num_generations = num_generations
    population_size = population_size
    remove_zero = not include_zero
    verbose = verbose
    if no_subs:
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
        gameweek_range,
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

    points = best_squad.get_expected_points(gameweek_start, tag)
    print("---------------------")
    print("Best expected points for gameweek {}: {}".format(gameweek_start, points))
    print("---------------------")
    print(best_squad)

    fpl_team_id = fpl_team_id or fetcher.FPL_TEAM_ID
    fill_initial_suggestion_table(
        best_squad,
        fpl_team_id,
        tag,
        season=season,
        gameweek=gameweek_start,
    )
