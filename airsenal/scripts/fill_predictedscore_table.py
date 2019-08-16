#!/usr/bin/env python

"""
Fill the "player_prediction" table with score predictions
Usage:
python fill_predictedscore_table.py --weeks_ahead <nweeks>
Generates a "tag" string which is stored so it can later be used by team-optimizers to
get consistent sets of predictions from the database.
"""
from uuid import uuid4

from multiprocessing import Process, Queue
from tqdm import tqdm
import argparse

from ..framework.utils import list_players, get_next_gameweek, CURRENT_SEASON
from ..framework.prediction_utils import get_fitted_team_model, get_fitted_player_model, \
    get_player_model, calc_predicted_points
from ..framework.schema import session_scope


def calc_predicted_points_for_pos(queue, gw_range, team_model, player_model, season, tag, session):
    """
    Calculate points predictions for all players in a given position and
    put into the DB
    """
    while True:
        pos = queue.get()
        print("Processing {}".format(pos))
        if pos == "DONE":
            print("Finished processing {}".format(pos))
            break
        predictions = {}
        df_player = None
        if pos != "GK": # don't calculate attacking points for keepers.
            df_player = get_fitted_player_model(player_model, pos, season, session)
        for player in list_players(position=pos, dbsession=session):
            predictions[player.player_id] = calc_predicted_points(
                player, team_model, df_player, season, tag, session, gw_range
            )
        ## commit changes to the db
        session.commit()
##    return predictions


def calc_all_predicted_points(gw_range, season, tag, session, num_thread=4):
    """
    Do the full prediction for players.
    """
    model_team = get_fitted_team_model(season, session)
    model_player = get_player_model()
    all_predictions = {}
    queue = Queue()
    procs = []
    for i in range(num_thread):
        processor = Process(
            target=calc_predicted_points_for_pos,
            args=(queue, gw_range, model_team, model_player, season, tag, session)
        )
        processor.daemon = True
        processor.start()
        procs.append(processor)


    for pos in ["GK", "DEF", "MID", "FWD"]:
        queue.put(pos)
    for i in range(num_thread):
        queue.put("DONE")

 #       all_predictions[pos] = calc_predicted_points_for_pos(pos,
 #                                                            gw_range,
 #                                                            model_team,
 #                                                            model_player,
 #                                                            season,
 #                                                            tag,
 #                                                            session)
  #  return all_predictions


def make_predictedscore_table(session, gw_range=None, season=CURRENT_SEASON, num_thread=4):
    tag = str(uuid4())
    if not gw_range:
        next_gameweek = get_next_gameweek()
        gw_range = list(range(next_gameweek, next_gameweek+3))
    prediction_dict = calc_all_predicted_points(gw_range, season, tag,  session)


def main():
    """
    fill the player_prediction db table
    """
    parser = argparse.ArgumentParser(description="fill player predictions")
    parser.add_argument(
        "--weeks_ahead", help="how many weeks ahead to fill", type=int
    )
    parser.add_argument(
        "--gameweek_start", help="first gameweek to look at", type=int
    )
    parser.add_argument(
        "--gameweek_end", help="last gameweek to look at", type=int
    )
    parser.add_argument(
        "--ep_filename", help="csv filename for FPL expected points"
    )
    parser.add_argument(
        "--season", help="season, in format e.g. '1819'",default=CURRENT_SEASON
    )
    parser.add_argument(
        "--num_thread", help="number of threads to parallelise over",default=4
    )
    args = parser.parse_args()
    if args.weeks_ahead and (args.gameweek_start or args.gameweek_end):
        print("Please specify either gameweek_start and gameweek_end, OR weeks_ahead")
        raise RuntimeError("Inconsistent arguments")
    if args.weeks_ahead and not args.season==CURRENT_SEASON:
        print("For past seasons, please specify gameweek_start and gameweek_end")
        raise RuntimeError("Inconsistent arguments")
    next_gameweek = get_next_gameweek()
    if args.weeks_ahead:
        gw_range = list(range(next_gameweek, next_gameweek+args.weeks_ahead))
    elif args.gameweek_start and args.gameweek_end:
        gw_range = list(range(args.gameweek_start, args.gameweek_end))
    elif args.gameweek_start:  # by default go three weeks ahead
        gw_range = list(range(args.gameweek_start, args.gameweek_start+3))
    else:
        gw_range = list(range(next_gameweek, next_gameweek+3))
    with session_scope() as session:
        make_predictedscore_table(session,
                                  gw_range=gw_range,
                                  season=args.season,
                                  num_thread = args.num_thread
        )
