"""
Functions to optimise an initial squad or a squad for wildcards/free hits.
"""

import random

from airsenal.framework.optimization_utils import get_discounted_squad_score, positions
from airsenal.framework.player import CandidatePlayer
from airsenal.framework.squad import TOTAL_PER_POSITION, Squad
from airsenal.framework.utils import CURRENT_SEASON, get_predicted_points


def make_new_squad(
    gw_range,
    tag,
    budget=1000,
    season=CURRENT_SEASON,
    verbose=1,
    bench_boost_gw=None,
    triple_captain_gw=None,
    algorithm="genetic",
    **kwargs,
):
    """
    Optimise a new squad from scratch with one of two algorithms:
    - algorithm="normal" : airsenal.framework.optimization_squad.make_new_squad_iter
    - algorithm="genetic": airsenal.framework.optimization_pygmo.make_new_squad_pygmo
    """
    if algorithm == "genetic":
        try:
            from airsenal.framework.optimization_pygmo import make_new_squad_pygmo

            return make_new_squad_pygmo(
                gw_range=gw_range,
                tag=tag,
                budget=budget,
                season=season,
                bench_boost_gw=bench_boost_gw,
                triple_captain_gw=triple_captain_gw,
                verbose=verbose,
                **kwargs,
            )
        except ModuleNotFoundError:
            print("Running optimisation without pygmo instead...")

    return make_new_squad_iter(
        gw_range=gw_range,
        tag=tag,
        budget=budget,
        season=season,
        bench_boost_gw=bench_boost_gw,
        triple_captain_gw=triple_captain_gw,
        verbose=verbose,
        **kwargs,
    )


def make_new_squad_iter(
    gw_range,
    tag,
    budget=1000,
    season=CURRENT_SEASON,
    num_iterations=100,
    update_func_and_args=None,
    verbose=False,
    bench_boost_gw=None,
    triple_captain_gw=None,
    **kwargs,
):
    """
    Make a squad from scratch, i.e. for gameweek 1, or for wildcard, or free hit, by
    selecting high scoring players and then iteratively replacing them with cheaper
    options until we have a valid squad.
    """
    transfer_gw = min(gw_range)  # the gw we're making the new squad
    best_score = 0.0
    best_squad = None

    for iteration in range(num_iterations):
        if verbose:
            print(f"Choosing new squad: iteration {iteration}")
        if update_func_and_args:
            # call function to update progress bar.
            # this was passed as a tuple (func, increment, pid)
            update_func_and_args[0](update_func_and_args[1], update_func_and_args[2])
        predicted_points = {}
        t = Squad(budget, season=season)
        # first iteration - fill up from the front
        for pos in positions:
            predicted_points[pos] = get_predicted_points(
                gameweek=gw_range, position=pos, tag=tag, season=season
            )
            for pp in predicted_points[pos]:
                t.add_player(pp[0], gameweek=transfer_gw)
                if t.num_position[pos] == TOTAL_PER_POSITION[pos]:
                    break

        # presumably we didn't get a complete squad now
        excluded_player_ids = []
        while not t.is_complete():
            # randomly swap out a player and replace with a cheaper one in the
            # same position
            player_to_remove = t.players[random.randint(0, len(t.players) - 1)]
            remove_cost = player_to_remove.purchase_price
            t.remove_player(player_to_remove.player_id, gameweek=transfer_gw)
            excluded_player_ids.append(player_to_remove.player_id)
            for pp in predicted_points[player_to_remove.position]:
                if (
                    pp[0] not in excluded_player_ids or random.random() < 0.3
                ):  # some chance to put player back
                    cp = CandidatePlayer(pp[0], gameweek=transfer_gw, season=season)
                    if cp.purchase_price >= remove_cost:
                        continue
                    else:
                        t.add_player(pp[0], gameweek=transfer_gw)
            # now try again to fill up the rest of the squad
            for pos in positions:
                num_missing = TOTAL_PER_POSITION[pos] - t.num_position[pos]
                if num_missing == 0:
                    continue
                for pp in predicted_points[pos]:
                    if pp[0] in excluded_player_ids:
                        continue
                    t.add_player(pp[0], gameweek=transfer_gw)
                    if t.num_position[pos] == TOTAL_PER_POSITION[pos]:
                        break
        # we have a complete squad
        score = get_discounted_squad_score(
            t,
            gw_range,
            tag,
            gw_range[0],
            bench_boost_gw,
            triple_captain_gw,
        )
        if score > best_score:
            best_score = score
            best_squad = t

    if verbose:
        print("====================================\n")
        print(best_squad)
        print(best_score)
    return best_squad
