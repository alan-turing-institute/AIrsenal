"""Fill the "fixture" table from this season's FPL API and past seasons' files."""

import uuid

from sqlalchemy.orm.session import Session

from airsenal.core.caching import clear_query_caches
from airsenal.core.console import track
from airsenal.core.data_files import FilePath, data_file
from airsenal.db.models import Fixture
from airsenal.db.queries.fixtures import find_fixture
from airsenal.db.session import get_session
from airsenal.game.mappings import canonical_team_name
from airsenal.game.season import CURRENT_SEASON, default_seasons, sort_seasons
from airsenal.remote.fpl_api import get_fetcher


def fill_fixtures_from_file(
    filename: FilePath, season: str, dbsession: Session | None = None
) -> None:
    """A season's matches, read from its results CSV file."""
    dbsession = get_session(dbsession)
    with open(filename) as infile:
        for line in track(infile.readlines()[1:], description=f"FIXTURES {season}"):
            fields = line.strip().split(",")
            f = Fixture()
            f.date = fields[0]
            f.gameweek = int(fields[5])
            # an unknown team is left unset
            if (home_team := canonical_team_name(fields[1])) is not None:
                f.home_team = home_team
            if (away_team := canonical_team_name(fields[2])) is not None:
                f.away_team = away_team
            f.season = season
            f.tag = "latest"  # not really needed for past seasons
            dbsession.add(f)
    dbsession.commit()


def fill_fixtures_from_api(season: str, dbsession: Session | None = None) -> None:
    """A season's fixtures, from the FPL API."""
    dbsession = get_session(dbsession)
    tag = str(uuid.uuid4())
    fetcher = get_fetcher()
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

        f.date = fixture["kickoff_time"]
        f.gameweek = fixture["event"]
        f.season = season
        f.tag = tag

        home_id = fixture["team_h"]
        away_id = fixture["team_a"]
        home_team = canonical_team_name(str(home_id))
        away_team = canonical_team_name(str(away_id))
        if home_team is None and away_team is None:
            msg = f"Can't find team(s) with id(s): {home_id}, {away_id}."
            raise ValueError(msg)
        if home_team is None:
            msg = f"Can't find team(s) with id(s): {home_id}"
            raise ValueError(msg)
        if away_team is None:
            msg = f"Can't find team(s) with id(s): {away_id}"
            raise ValueError(msg)
        f.home_team = home_team
        f.away_team = away_team
        dbsession.add(f)
    dbsession.commit()


def make_fixture_table(
    seasons: list[str] | None = None, dbsession: Session | None = None
) -> None:
    dbsession = get_session(dbsession)
    if not seasons:
        seasons = default_seasons()
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
