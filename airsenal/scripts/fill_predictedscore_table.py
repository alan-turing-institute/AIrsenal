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
import argparse

from ..framework.utils import list_players, get_next_gameweek, CURRENT_SEASON
from ..framework.prediction_utils import get_fitted_models, calc_predicted_points
from ..framework.schema import session_scope

def calc_all_predicted_points(gw_range, season, tag, session):
    """
    Do the full prediction.
    """
    model_team, df_player = get_fitted_models(season, session)
    all_predictions = {}
    for pos in ["GK", "DEF", "MID", "FWD"]:
        for player in list_players(position=pos, dbsession=session):
            all_predictions[player.player_id] = calc_predicted_points(
                player, model_team, df_player, season, tag, session, gw_range
            )
    ## commit changes to the db
    session.commit()
    return all_predictions

def make_predictedscore_table(session, gw_range=None, season=CURRENT_SEASON):
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
                                  season=args.season)
