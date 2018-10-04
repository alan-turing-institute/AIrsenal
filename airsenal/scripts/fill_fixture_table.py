#!/usr/bin/env python

"""
Fill the "fixture" table with info from this seasons FPL
(fixtures.csv).
"""

import os
import dateparser
import time
import uuid

from ..framework.data_fetcher import FPLDataFetcher, MatchDataFetcher
from ..framework.mappings import alternative_team_names
from ..framework.schema import Fixture, session_scope




def fill_fixtures_from_file(filename, season, session):
    """
    use the match results csv files to get a list of matches in a season,
    """
    infile = open(filename)
    for line in infile.readlines()[1:]:
        fields = line.strip().split(",")
        f = Fixture()
        f.date = fields[0]
        f.gameweek = fields[5]
        home_team = fields[1]
        away_team = fields[2]
        for k, v in alternative_team_names.items():
            if home_team in v:
                f.home_team = k
            elif away_team in v:
                f.away_team = k
        print(" ==> Filling fixture {} {}".format(f.home_team, f.away_team))
        f.season = season
        f.tag = "latest" # not really needed for past seasons
        session.add(f)
    session.commit()


def fill_fixtures_from_api(season, session):
    """
    Use the football data api to get a list of matches, and the FPL
    api to get deadlines.
    """
    tag = str(uuid.uuid4())
    fetcher = FPLDataFetcher()
    deadlines = fetcher.get_event_data()
    mf = MatchDataFetcher()
    all_matches = []
    for gw in range(1,39):
        if gw%10 == 0:
            time.sleep(60) # can only hit API 10 times per minute
        all_matches += mf.get_fixtures(gw)
    ## now loop over all matches, and see what is the first deadline after
    ## the match date
    for match in all_matches:
        match_time = dateparser.parse(match[0])
        gameweek = None
        for gw_idx in range(1,38):
            this_deadline = dateparser.parse(deadlines[gw_idx]['deadline'])
            next_deadline = dateparser.parse(deadlines[gw_idx+1]['deadline'])
            if match_time > this_deadline and match_time < next_deadline:
                gameweek = gw_idx
                break
        if not gameweek: # must be the last gameweek
            gameweek = 38
        print("match {} {} is gameweek {}".format(match[1],
                                                  match[2],
                                                  gameweek))
        f = Fixture()
        f.date = match[0]
        f.home_team = match[1]
        f.away_team = match[2]
        f.gameweek = gameweek
        f.season = season
        f.tag = tag
        session.add(f)
    session.commit()
    return True


def make_fixture_table(session):
    # fill the fixture table for past seasons
    for season in ["1516","1617","1718"]:
        filename = os.path.join(os.path.dirname(__file__),
                                "..",
                                "data",
                                "results_{}_with_gw.csv".format(season))
        fill_fixtures_from_file(filename,season,session)
    # now fill the current season from the api
    fill_fixtures_from_api("1819",session)


if __name__ == "__main__":
    with session_scope() as session:
        make_fixture_table(session)
