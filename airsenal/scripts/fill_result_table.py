#!/usr/bin/env python

"""
Fill the "result" table with historic results
(results_xxyy_with_gw.csv).
"""

import argparse
import os

from ..framework.mappings import alternative_team_names
from ..framework.schema import Result, session_scope, Fixture
from ..framework.data_fetcher import FPLDataFetcher
from ..framework.utils import (
    NEXT_GAMEWEEK,
    get_latest_fixture_tag,
    get_past_seasons,
    CURRENT_SEASON,
)


def _find_fixture(season, home_team, away_team, session):
    """
    query database to find corresponding fixture
    """
    tag = get_latest_fixture_tag(season)
    f = (
        session.query(Fixture)
        .filter_by(tag=tag)
        .filter_by(season=season)
        .filter_by(home_team=home_team)
        .filter_by(away_team=away_team)
        .first()
    )
    return f


def fill_results_from_csv(input_file, season, session):
    for line in input_file.readlines()[1:]:
        (
            date,
            home_team,
            away_team,
            home_score,
            away_score,
            gameweek,
        ) = line.strip().split(",")
        print(line.strip())
        for k, v in alternative_team_names.items():
            if home_team in v:
                home_team = k
            elif away_team in v:
                away_team = k
        ## query database to find corresponding fixture
        tag = get_latest_fixture_tag(season, session)
        f = _find_fixture(season, home_team, away_team, session)
        res = Result()
        res.fixture = f
        res.home_score = int(home_score)
        res.away_score = int(away_score)
        session.add(res)
    session.commit()


def fill_results_from_api(gw_start, gw_end, season, session):
    fetcher = FPLDataFetcher()
    matches = fetcher.get_fixture_data()
    for m in matches:
        if not m["finished"]:
            continue
        gameweek = m["event"]
        if gameweek < gw_start or gameweek > gw_end:
            continue
        home_id = m["team_h"]
        away_id = m["team_a"]
        home_team = None
        away_team = None
        for k, v in alternative_team_names.items():
            if str(home_id) in v:
                home_team = k
            elif str(away_id) in v:
                away_team = k
        if not home_team:
            raise ValueError("Unable to find team with id {}".format(home_id))
        if not away_team:
            raise ValueError("Unable to find team with id {}".format(away_id))
        home_score = m["team_h_score"]
        away_score = m["team_a_score"]
        f = _find_fixture(season, home_team, away_team, session)
        res = Result()
        res.fixture = f
        res.home_score = int(home_score)
        res.away_score = int(away_score)
        session.add(res)
    session.commit()


def make_result_table(session):
    """
    past seasons - read results from csv
    """
    for season in get_past_seasons(3):
        inpath = os.path.join(
            os.path.dirname(__file__), "../data/results_{}_with_gw.csv".format(season)
        )
        infile = open(inpath)
        fill_results_from_csv(infile, season, session)
    """
    current season - use API
    """
    gw_end = NEXT_GAMEWEEK
    fill_results_from_api(1, gw_end, CURRENT_SEASON, session)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="fill table of match results")
    parser.add_argument("--input_type", help="csv or api", default="csv")
    parser.add_argument("--input_file", help="input csv filename")
    parser.add_argument(
        "--season",
        help="if using a single csv, specify the season",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--gw_start", help="if using api, which gameweeks", type=int, default=1
    )
    parser.add_argument("--gw_end", help="if using api, which gameweeks", type=int)
    args = parser.parse_args()

    with session_scope() as session:
        make_result_table(session)
