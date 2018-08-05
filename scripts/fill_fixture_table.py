#!/usr/bin/env python

"""
Fill the "fixture" table with info from this seasons FPL
(fixtures.csv).
"""

import os
import sys
sys.path.append("..")

import json

from data.mappings import alternative_team_names

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from framework.schema import Fixture, Base, engine

DBSession = sessionmaker(bind=engine)
session = DBSession()

if __name__ == "__main__":

    input_file = open("../data/fixtures.csv")
    for line in input_file.readlines()[1:]:
        gameweek, date, home_team, away_team = \
                        line.strip().split(",")
        print(line.strip())
        f = Fixture()
        f.gameweek = int(gameweek)
        f.date = date
        for k, v in alternative_team_names.items():
            if home_team in v:
                f.home_team = k
            elif away_team in v:
                f.away_team = k
        session.add(f)
    session.commit()
