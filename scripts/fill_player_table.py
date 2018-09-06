#!/usr/bin/env python

"""
Fill the "Player" table with info from this seasons FPL
(FPL_2017-18.json).
"""

import os
import sys

sys.path.append("..")

import json

from framework.mappings import alternative_team_names, positions

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from framework.schema import Player, Base, engine
from framework.data_fetcher import DataFetcher

DBSession = sessionmaker(bind=engine)
session = DBSession()

if __name__ == "__main__":
    df = DataFetcher()
    pd = df.get_player_summary_data()

    for k, v in pd.items():
        p = Player()
        p.player_id = k
        name = "{} {}".format(v["first_name"], v["second_name"])
        print("Adding {}".format(name))
        p.name = name
        team_number = v["team"]
        for tk, tv in alternative_team_names.items():
            if str(team_number) in tv:
                p.team = tk
                break
        p.position = positions[v["element_type"]]
        p.current_price = v["now_cost"]
        session.add(p)
    session.commit()
