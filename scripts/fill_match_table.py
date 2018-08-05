#!/usr/bin/env python

"""
Fill the "match" table with historic results
(results_xxyy_with_gw.csv).
"""

import os
import sys
sys.path.append("..")

import json

from data.mappings import alternative_team_names, positions

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from framework.schema import Match, Base, engine

DBSession = sessionmaker(bind=engine)
session = DBSession()

if __name__ == "__main__":
    for season in ["1718","1617","1516"]:
        input_file = open("../data/results_{}_with_gw.csv".format(season))
        for line in input_file.readlines()[1:]:
            date, home_team, away_team, home_score, away_score, gameweek = \
                        line.strip().split(",")
            print(line.strip())
            m = Match()
            m.season = season
            m.date = date
            m.home_score = int(home_score)
            m.away_score = int(away_score)
            m.gameweek = int(gameweek)
            for k, v in alternative_team_names.items():
                if home_team in v:
                    m.home_team = k
                elif away_team in v:
                    m.away_team = k
            session.add(m)
    session.commit()
