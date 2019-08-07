#!/usr/bin/env python

"""
Fill the "match" table with historic results
(results_xxyy_with_gw.csv).
"""

import argparse
import os

from ..framework.mappings import alternative_team_names
from ..framework.schema import Match, session_scope, Fixture
from ..framework.data_fetcher import MatchDataFetcher
from ..framework.utils import get_next_gameweek, get_latest_fixture_tag, \
    get_past_seasons, CURRENT_SEASON


def _find_fixture(season,home_team,away_team,session):
    """
    query database to find corresponding fixture
    """
    tag = get_latest_fixture_tag(season)
    f = session.query(Fixture)\
               .filter_by(tag=tag)\
               .filter_by(season=season)\
               .filter_by(home_team=home_team)\
               .filter_by(away_team=away_team)\
               .first()
    return f


def fill_table_from_csv(input_file, season, session):
    for line in input_file.readlines()[1:]:
        date, home_team, away_team, home_score, away_score, gameweek = line.strip().split(
            ","
        )
        print(line.strip())
        for k, v in alternative_team_names.items():
            if home_team in v:
                home_team = k
            elif away_team in v:
                away_team = k
        ## query database to find corresponding fixture
        tag = get_latest_fixture_tag(season)
        f = _find_fixture(season, home_team, away_team, session)
        m = Match()
        m.fixture = f
        m.home_score = int(home_score)
        m.away_score = int(away_score)
        session.add(m)
    session.commit()


def fill_table_from_list(input_list, gameweek, season, session):
    for result in input_list:
        print(result)
        date, home_team, away_team, home_score, away_score = result
        f = _find_fixture(season, home_team, away_team, session)
        m = Match()
        m.fixture = f
        m.home_score = int(home_score)
        m.away_score = int(away_score)
        session.add(m)


def fill_from_api(gw_start, gw_end, season, session):
    mf = MatchDataFetcher()
    for gw in range(gw_start, gw_end):
        results = mf.get_results(gw)
        fill_table_from_list(results, gw, season, session)
    session.commit()


def make_match_table(session):
    """
    past seasons - read results from csv
    """
    for season in get_past_seasons(3):
        inpath = os.path.join(os.path.dirname(__file__), "../data/results_{}_with_gw.csv".format(season))
        infile = open(inpath)
        fill_table_from_csv(infile, season, session)
    """
    current season - use API
    """
    gw_end = get_next_gameweek()
    fill_from_api(1, gw_end, CURRENT_SEASON, session)


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
        make_match_table(session)
