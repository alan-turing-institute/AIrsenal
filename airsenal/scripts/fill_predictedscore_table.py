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

from ..framework.utils import list_players
from ..framework.prediction_utils import get_fitted_models, get_predicted_points
from ..framework.schema import session_scope

def calc_all_predicted_points(weeks_ahead, season, tag, session):
    """
    Do the full prediction.
    """
    model_team, df_player = get_fitted_models(session)
    all_predictions = {}
    for pos in ["GK", "DEF", "MID", "FWD"]:
        for player in list_players(position=pos, dbsession=session):
            all_predictions[player.player_id] = get_predicted_points(
                player, model_team, df_player, season, tag, session, weeks_ahead
            )
    ## commit changes to the db
    session.commit()
    return all_predictions

def make_predictedscore_table(session, weeks_ahead=3, season="1819"):
    tag = str(uuid4())
    prediction_dict = calc_all_predicted_points(weeks_ahead, season, tag,  session)


if __name__ == "__main__":
    """
    fill the player_prediction db table
    """
    parser = argparse.ArgumentParser(description="fill player predictions")
    parser.add_argument(
        "--weeks_ahead", help="how many weeks ahead to fill", type=int, default=5
    )
    parser.add_argument(
        "--ep_filename", help="csv filename for FPL expected points"
    )
    args = parser.parse_args()

    with session_scope() as session:
        make_predictedscore_table(session, weeks_ahead=3)
