#!/usr/bin/env python

"""
Fill the "player_prediction" table with score predictions
"""

import os
import sys
sys.path.append("..")

import json

from framework.mappings import alternative_team_names, \
    alternative_player_names, positions

from sqlalchemy import create_engine, and_, or_
from sqlalchemy.orm import sessionmaker

from framework.schema import Player, PlayerPrediction, Fixture, Base, engine

from framework.data_fetcher import DataFetcher
from framework.utils import get_fixtures_for_player

DBSession = sessionmaker(bind=engine)
session = DBSession()


if __name__ == "__main__":

    if sys.argv[-1] == "EP":
        df = DataFetcher()
        playerdata = df.get_current_player_data()
        for k,v in playerdata.items():
            next_fixture = get_fixtures_for_player(k)[0]
            expected_points = v['ep_next']
            pp = PlayerPrediction()
            pp.player_id = k
            pp.fixture_id = next_fixture
            pp.predicted_points = float(expected_points)
            pp.method = "EP"
            session.add(pp)
        session.commit()
    else:
        print("Unknown method")
