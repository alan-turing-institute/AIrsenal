#!/usr/bin/env python

"""
Fill the "fixture" table with info from this seasons FPL
(fixtures.csv).
"""

import os
import uuid
from typing import List, Optional

from sqlalchemy.orm.session import Session

from airsenal.framework.data_fetcher import FPLDataFetcher
from airsenal.framework.mappings import alternative_team_names
from airsenal.framework.schema import Fixture, session, session_scope
from airsenal.framework.season import CURRENT_SEASON, sort_seasons
from airsenal.framework.utils import find_fixture, get_past_seasons


def fill_fixtures_from_file(
    filename: str, season: str, dbsession: Session = session
) -> None:
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
        print(f" ==> Filling fixture {f.home_team} {f.away_team}")
        f.season = season
        f.tag = "latest"  # not really needed for past seasons
        dbsession.add(f)
    dbsession.commit()


def fill_fixtures_from_api(season: str, dbsession: Session = session) -> None:
    """
    Use the FPL API to get a list of fixures.
    """
    tag = str(uuid.uuid4())
    fetcher = FPLDataFetcher()
    fixtures = fetcher.get_fixture_data()
    for fixture in fixtures:
        try:
            f = find_fixture(
                fixture["team_h"],
                was_home=True,
                other_team=fixture["team_a"],
                season=season,
                dbsession=dbsession,
            )
            update = True
        except ValueError:
            f = Fixture()
            update = False

        f.date = fixture["kickoff_time"]
        f.gameweek = fixture["event"]
        f.season = season
        f.tag = tag

        home_id = fixture["team_h"]
        away_id = fixture["team_a"]
        found_home = False
        found_away = False
        for k, v in alternative_team_names.items():
            if str(home_id) in v:
                f.home_team = k
                found_home = True
            elif str(away_id) in v:
                f.away_team = k
                found_away = True
            if found_home and found_away:
                break

        if not found_home and found_away:
            raise ValueError(f"Can't find team(s) with id(s): {home_id}, {away_id}.")
        elif not found_home:
            raise ValueError(f"Can't find team(s) with id(s): {home_id}")
        elif not found_away:
            raise ValueError(f"Can't find team(s) with id(s): {away_id}")
        if not update:
            dbsession.add(f)
    dbsession.commit()


def make_fixture_table(
    seasons: Optional[List[str]] = [], dbsession: Session = session
) -> None:
    # fill the fixture table for past seasons
    if not seasons:
        seasons = [CURRENT_SEASON]
        seasons += get_past_seasons(3)
    for season in sort_seasons(seasons):
        if season == CURRENT_SEASON:
            # current season - use API
            fill_fixtures_from_api(CURRENT_SEASON, dbsession=dbsession)
        else:
            filename = os.path.join(
                os.path.dirname(__file__),
                "..",
                "data",
                f"results_{season}_with_gw.csv",
            )
            fill_fixtures_from_file(filename, season, dbsession=dbsession)


if __name__ == "__main__":
    with session_scope() as session:
        make_fixture_table(dbsession=session)
