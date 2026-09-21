"""
Fill the "fixture" table with info from this seasons FPL (fixtures.csv).
"""

import os
import uuid

from sqlalchemy.orm.session import Session

from sqlalchemy import select

from airsenal.framework.data_fetcher import FPLDataFetcher
from airsenal.framework.mappings import alternative_team_names
from airsenal.framework.schema import Fixture, Team, session, session_scope
from airsenal.framework.season import CURRENT_SEASON, sort_seasons
from airsenal.framework.utils import find_fixture, get_past_seasons


def fill_fixtures_from_file(
    filename: str, season: str, dbsession: Session = session
) -> None:
    """
    use the match results csv files to get a list of matches in a season,
    """
    with open(filename) as infile:
        for line in infile.readlines()[1:]:
            fields = line.strip().split(",")
            f = Fixture()
            f.date = fields[0]
            f.gameweek = int(fields[5])
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

    The FPL team_id is reassigned between seasons: the id that meant Burnley
    a few years back now means Bournemouth. alternative_team_names carries
    those historical id->name entries and so cannot be trusted to translate
    a current-season fixture. The Team table was just populated live from
    the same API, so use that instead.
    """
    tag = str(uuid.uuid4())
    fetcher = FPLDataFetcher()
    fixtures = fetcher.get_fixture_data()
    id_to_name = {
        t.team_id: t.name
        for t in dbsession.scalars(select(Team).where(Team.season == season)).all()
    }
    if not id_to_name:
        msg = (
            f"No teams in the DB for {season} - fill the Team table "
            "before filling fixtures"
        )
        raise RuntimeError(msg)

    for fixture in fixtures:
        f = find_fixture(
            fixture["team_h"],
            was_home=True,
            other_team=fixture["team_a"],
            season=season,
            dbsession=dbsession,
        )
        if f is None:
            print("Creating new fixture")
            f = Fixture()
            update = False
        else:
            update = True

        f.date = fixture["kickoff_time"]
        f.gameweek = fixture["event"]
        f.season = season
        f.tag = tag

        home_id = fixture["team_h"]
        away_id = fixture["team_a"]
        home_name = id_to_name.get(home_id)
        away_name = id_to_name.get(away_id)
        if not home_name and not away_name:
            msg = f"Can't find team(s) with id(s): {home_id}, {away_id}."
            raise ValueError(msg)
        if not home_name:
            msg = f"Can't find team(s) with id(s): {home_id}"
            raise ValueError(msg)
        if not away_name:
            msg = f"Can't find team(s) with id(s): {away_id}"
            raise ValueError(msg)
        f.home_team = home_name
        f.away_team = away_name
        if not update:
            dbsession.add(f)
    dbsession.commit()


def make_fixture_table(
    seasons: list[str] | None = None, dbsession: Session = session
) -> None:
    # fill the fixture table for past seasons
    if seasons is None:
        seasons = []
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
                f"results_{season}.csv",
            )
            fill_fixtures_from_file(filename, season, dbsession=dbsession)


if __name__ == "__main__":
    with session_scope() as session:
        make_fixture_table(dbsession=session)
