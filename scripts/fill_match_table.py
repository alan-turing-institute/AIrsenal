#!/usr/bin/env python

"""
Fill the "match" table with historic results
(results_xxyy_with_gw.csv).
"""

import os
import sys
sys.path.append("..")
import argparse
import json

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from framework.mappings import alternative_team_names, positions
from framework.schema import Match, Base, engine
from framework.data_fetcher import MatchDataFetcher

DBSession = sessionmaker(bind=engine)
session = DBSession()

def fill_table_from_csv(input_file, season):
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

def fill_table_from_list(input_list, gameweek):
    for result in input_list:
        print(result)
        date, home_team, away_team, home_score, away_score = result
        m = Match()
        m.season = "1819"
        m.date = date
        m.home_team = home_team
        m.away_team = away_team
        m.home_score = int(home_score)
        m.away_score = int(away_score)
        m.gameweek = int(gameweek)
        session.add(m)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="fill table of match results")
    parser.add_argument("--input_type",help="csv or api", default="csv")
    parser.add_argument("--input_file",help="input csv filename")
    parser.add_argument("--gw_start",help="if using api, which gameweeks",
                        type=int, default=0)
    parser.add_argument("--gw_end",help="if using api, which gameweeks",
                        type=int, default=39)
    args = parser.parse_args()
    if args.input_type == "csv":
        if args.input_file:
            infile = open(input_file)
            fill_table_from_csv(infile)
        else:
            for season in ["1819","1718","1617","1516"]:
                infile = open("../data/results_{}_with_gw.csv".format(season))
                fill_table_from_csv(infile, season)
    else:
        ## use the API
        mf = MatchDataFetcher()
        for gw in range(args.gw_start, args.gw_end):
            results = mf.get_results(gw)
            fill_table_from_list(results,gw)
    session.commit()
