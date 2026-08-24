"""
Fill the "fixture" table with info from this seasons FPL (fixtures.csv).
"""

import uuid

from sqlalchemy.orm.session import Session

from airsenal.core.caching import clear_query_caches
from airsenal.core.console import track
from airsenal.core.data_files import FilePath, data_file
from airsenal.core.mappings import alternative_team_names
from airsenal.core.season import CURRENT_SEASON, get_past_seasons, sort_seasons
from airsenal.db.models import Fixture
from airsenal.db.queries.fixtures import find_fixture
from airsenal.db.session import get_session, session_scope
from airsenal.fetch.fpl_api import FPLDataFetcher


def fill_fixtures_from_file(
    filename: FilePath, season: str, dbsession: Session | None = None
) -> None:
    """
    use the match results csv files to get a list of matches in a season,
    """
    dbsession = dbsession if dbsession is not None else get_session()
    with open(filename) as infile:
        for line in track(infile.readlines()[1:], description=f"FIXTURES {season}"):
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
            f.season = season
            f.tag = "latest"  # not really needed for past seasons
            dbsession.add(f)
    dbsession.commit()


def fill_fixtures_from_api(season: str, dbsession: Session | None = None) -> None:
    """
    Use the FPL API to get a list of fixures.
    """
    dbsession = dbsession if dbsession is not None else get_session()
    tag = str(uuid.uuid4())
    fetcher = FPLDataFetcher()
    fixtures = fetcher.get_fixture_data()
    for fixture in track(fixtures, description=f"FIXTURES {season}"):
        f = find_fixture(
            fixture["team_h"],
            was_home=True,
            other_team=fixture["team_a"],
            season=season,
            dbsession=dbsession,
            verbose=False,
        )
        if f is None:
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
            msg = f"Can't find team(s) with id(s): {home_id}, {away_id}."
            raise ValueError(msg)
        if not found_home:
            msg = f"Can't find team(s) with id(s): {home_id}"
            raise ValueError(msg)
        if not found_away:
            msg = f"Can't find team(s) with id(s): {away_id}"
            raise ValueError(msg)
        if not update:
            dbsession.add(f)
    dbsession.commit()


def make_fixture_table(
    seasons: list[str] | None = None, dbsession: Session | None = None
) -> None:
    # fill the fixture table for past seasons
    dbsession = dbsession if dbsession is not None else get_session()
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
            fill_fixtures_from_file(
                data_file(f"results_{season}.csv"), season, dbsession=dbsession
            )
    # gameweek lookups are cached, and every one of them reads this table
    clear_query_caches()


if __name__ == "__main__":
    with session_scope() as session:
        make_fixture_table(dbsession=session)
