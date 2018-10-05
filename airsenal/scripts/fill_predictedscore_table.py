#!/usr/bin/env python

"""
Fill the "player_prediction" table with score predictions
Usage:
python fill_predictedscore_table.py --weeks_ahead <nweeks>
Generates a "tag" string which is stored so it can later be used by team-optimizers to
get consistent sets of predictions from the database.
"""
from uuid import uuid4

import argparse

from ..framework.prediction_utils import calc_all_predicted_points, fill_table
from ..framework.schema import session_scope


def make_predictedscore_table(session, weeks_ahead=3):
    prediction_dict = calc_all_predicted_points(weeks_ahead, session)
    fill_table(prediction_dict, str(uuid4()), session)



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
