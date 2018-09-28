#!/usr/bin/env python

"""
Fill the "match" table with historic results
(results_xxyy_with_gw.csv).
"""

import argparse
import os

from ..framework.mappings import alternative_team_names
from ..framework.schema import Match, session_scope
from ..framework.data_fetcher import MatchDataFetcher
from ..framework.utils import get_next_gameweek


def fill_table_from_csv(input_file, season, session):
    for line in input_file.readlines()[1:]:
        date, home_team, away_team, home_score, away_score, gameweek = line.strip().split(
            ","
        )
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


def fill_table_from_list(input_list, gameweek, session):
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


def fill_from_api(gw_start, gw_end, session):
    mf = MatchDataFetcher()
    for gw in range(gw_start, gw_end):
        results = mf.get_results(gw)
        fill_table_from_list(results, gw)
    session.commit()


def make_match_table(input_type, session, gw_start=None, gw_end=None, season=None, input_file=None):
    if input_type == "csv":
        if input_file:
            if season is None:
                raise ValueError("If using a specified csv input file, you must provide a season identifier.")
            infile = open(input_file)
            fill_table_from_csv(infile, season, session)
        else:
            for season in ["1718", "1617", "1516"]:
                inpath = os.path.join(os.path.dirname(__file__), "../data/results_{}_with_gw.csv".format(season))
                infile = open(inpath)
                fill_table_from_csv(infile, season, session)
    else:
        # use the API
        if gw_start is None:
            raise ValueError("Must provide a starting gameweek if using the API.")
        if not gw_end:
            gw_end = get_next_gameweek()
        else:
            gw_end = gw_end
        fill_from_api(gw_start, gw_end, session)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="fill table of match results")
    parser.add_argument("--input_type", help="csv or api", default="csv")
    parser.add_argument("--input_file", help="input csv filename")
    parser.add_argument("--season", help="if using a single csv, specify the season", type=str, default=None)
    parser.add_argument(
        "--gw_start", help="if using api, which gameweeks", type=int, default=1
    )
    parser.add_argument("--gw_end", help="if using api, which gameweeks", type=int)
    args = parser.parse_args()

    with session_scope() as session:
        make_match_table(args.input_type,
                         session,
                         gw_start=args.gw_start,
                         gw_end=args.gw_end,
                         season=args.season,
                         input_file=args.input_file)

