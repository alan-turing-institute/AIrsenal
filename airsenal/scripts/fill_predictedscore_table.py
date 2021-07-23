#!/usr/bin/env python

"""
Fill the "player_prediction" table with score predictions
Usage:
python fill_predictedscore_table.py --weeks_ahead <nweeks>
Generates a "tag" string which is stored so it can later be used by team-optimizers to
get consistent sets of predictions from the database.
"""
from uuid import uuid4
import pickle
import pkg_resources

from multiprocessing import Process, Queue
import argparse

from airsenal.framework.bpl_interface import get_fitted_team_model
from airsenal.framework.utils import (
    NEXT_GAMEWEEK,
    CURRENT_SEASON,
    get_top_predicted_points,
    list_players,
)

from airsenal.framework.prediction_utils import (
    calc_predicted_points_for_player,
    get_all_fitted_player_models,
    fit_bonus_points,
    fit_save_points,
    fit_card_points,
)

from airsenal.framework.schema import session_scope


def allocate_predictions(
    queue,
    gw_range,
    team_model,
    df_player,
    df_bonus,
    df_saves,
    df_cards,
    season,
    tag,
    dbsession,
):
    """
    Take positions off the queue and call function to calculate predictions
    """
    while True:
        player = queue.get()
        if player == "DONE":
            print("Finished processing")
            break

        predictions = calc_predicted_points_for_player(
            player,
            team_model,
            df_player,
            df_bonus,
            df_saves,
            df_cards,
            season,
            gw_range=gw_range,
            tag=tag,
            dbsession=dbsession,
        )
        for p in predictions:
            dbsession.add(p)
        dbsession.commit()


def calc_all_predicted_points(
    gw_range,
    season,
    include_bonus=True,
    include_cards=True,
    include_saves=True,
    num_thread=4,
    tag="",
    dbsession=None,
):
    """
    Do the full prediction for players.
    """
    model_team = get_fitted_team_model(
        season, gameweek=min(gw_range), dbsession=dbsession
    )
    model_file = pkg_resources.resource_filename(
        "airsenal", "stan_model/player_forecasts.pkl"
    )
    print("Loading pre-compiled player model from {}".format(model_file))
    with open(model_file, "rb") as f:
        model_player = pickle.load(f)

    df_player = get_all_fitted_player_models(model_player, season, gw_range[0])

    if include_bonus:
        df_bonus = fit_bonus_points(gameweek=gw_range[0], season=season)
    else:
        df_bonus = None
    if include_saves:
        df_saves = fit_save_points(gameweek=gw_range[0], season=season)
    else:
        df_saves = None
    if include_cards:
        df_cards = fit_card_points(gameweek=gw_range[0], season=season)
    else:
        df_cards = None

    players = list_players(season=season, gameweek=gw_range[0], dbsession=dbsession)

    if num_thread is not None and num_thread > 1:
        queue = Queue()
        procs = []
        for _ in range(num_thread):
            processor = Process(
                target=allocate_predictions,
                args=(
                    queue,
                    gw_range,
                    model_team,
                    df_player,
                    df_bonus,
                    df_saves,
                    df_cards,
                    season,
                    tag,
                    dbsession,
                ),
            )
            processor.daemon = True
            processor.start()
            procs.append(processor)

        for p in players:
            queue.put(p.player_id)
        for _ in range(num_thread):
            queue.put("DONE")

        for _, p in enumerate(procs):
            p.join()
    else:
        # single threaded
        for player in players:
            predictions = calc_predicted_points_for_player(
                player,
                model_team,
                df_player,
                df_bonus,
                df_saves,
                df_cards,
                season,
                gw_range=gw_range,
                tag=tag,
                dbsession=dbsession,
            )
            for p in predictions:
                dbsession.add(p)
        dbsession.commit()
        print("Finished adding predictions to db")


def make_predictedscore_table(
    gw_range=None,
    season=CURRENT_SEASON,
    num_thread=4,
    include_bonus=True,
    include_cards=True,
    include_saves=True,
    tag_prefix=None,
    dbsession=None,
):
    tag = tag_prefix or ""
    tag += str(uuid4())
    if not gw_range:
        gw_range = list(range(NEXT_GAMEWEEK, NEXT_GAMEWEEK + 3))
    calc_all_predicted_points(
        gw_range,
        season,
        include_bonus=include_bonus,
        include_cards=include_cards,
        include_saves=include_saves,
        num_thread=num_thread,
        tag=tag,
        dbsession=dbsession,
    )
    return tag


def main():
    """
    fill the player_prediction db table
    """
    parser = argparse.ArgumentParser(description="fill player predictions")
    parser.add_argument("--weeks_ahead", help="how many weeks ahead to fill", type=int)
    parser.add_argument("--gameweek_start", help="first gameweek to look at", type=int)
    parser.add_argument("--gameweek_end", help="last gameweek to look at", type=int)
    parser.add_argument("--ep_filename", help="csv filename for FPL expected points")
    parser.add_argument(
        "--season", help="season, in format e.g. '1819'", default=CURRENT_SEASON
    )
    parser.add_argument(
        "--num_thread", help="number of threads to parallelise over", type=int
    )
    parser.add_argument(
        "--no_bonus",
        help="don't include bonus points",
        action="store_true",
    )
    parser.add_argument(
        "--no_cards",
        help="don't include points lost to yellow and red cards",
        action="store_true",
    )
    parser.add_argument(
        "--no_saves",
        help="don't include save points for goalkeepers",
        action="store_true",
    )

    args = parser.parse_args()
    if args.weeks_ahead and (args.gameweek_start or args.gameweek_end):
        print("Please specify either gameweek_start and gameweek_end, OR weeks_ahead")
        raise RuntimeError("Inconsistent arguments")
    if args.weeks_ahead and args.season != CURRENT_SEASON:
        print("For past seasons, please specify gameweek_start and gameweek_end")
        raise RuntimeError("Inconsistent arguments")
    if args.weeks_ahead:
        gw_range = list(range(NEXT_GAMEWEEK, NEXT_GAMEWEEK + args.weeks_ahead))
    elif args.gameweek_start and args.gameweek_end:
        gw_range = list(range(args.gameweek_start, args.gameweek_end))
    elif args.gameweek_start:  # by default go three weeks ahead
        gw_range = list(range(args.gameweek_start, args.gameweek_start + 3))
    else:
        gw_range = list(range(NEXT_GAMEWEEK, NEXT_GAMEWEEK + 3))
    num_thread = args.num_thread or None
    include_bonus = not args.no_bonus
    include_cards = not args.no_cards
    include_saves = not args.no_saves

    with session_scope() as session:
        session.expire_on_commit = False

        tag = make_predictedscore_table(
            gw_range=gw_range,
            season=args.season,
            num_thread=num_thread,
            include_bonus=include_bonus,
            include_cards=include_cards,
            include_saves=include_saves,
            dbsession=session,
        )

        # print players with top predicted points
        get_top_predicted_points(
            gameweek=gw_range,
            tag=tag,
            season=args.season,
            dbsession=session,
            per_position=True,
            n_players=5,
        )
