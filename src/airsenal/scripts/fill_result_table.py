"""
Fill the "result" table with historic results (results_xxyy_with_gw.csv).
"""

from sqlalchemy.orm.session import Session

from airsenal.core.console import track
from airsenal.core.logging import get_logger
from airsenal.core.resources import FilePath, resource
from airsenal.db.models import Result
from airsenal.db.queries.fixtures import find_fixture
from airsenal.db.queries.gameweeks import (
    get_last_complete_gameweek_in_db,
    next_gameweek,
)
from airsenal.db.session import get_session
from airsenal.domain.mappings import alternative_team_names
from airsenal.domain.season import CURRENT_SEASON, get_past_seasons, sort_seasons
from airsenal.fetch.fpl_api import FPLDataFetcher, get_fetcher
from airsenal.fetch.gameweeks import get_last_finished_gameweek

logger = get_logger(__name__)


def fill_results_from_csv(
    input_file: FilePath, season: str, dbsession: Session
) -> None:
    with open(input_file) as f:
        lines = f.readlines()
    for line in track(lines[1:], description=f"RESULTS {season}"):
        (
            _date,
            home_team,
            away_team,
            home_score,
            away_score,
            _gameweek,
        ) = line.strip().split(",")
        for k, v in alternative_team_names.items():
            if home_team in v:
                home_team = k
            elif away_team in v:
                away_team = k
        # query database to find corresponding fixture
        fixture = find_fixture(
            home_team,
            was_home=True,
            other_team=away_team,
            season=season,
            dbsession=dbsession,
        )
        if fixture is None:
            logger.warning(
                "Unable to find fixture for %s vs %s in %s",
                home_team,
                away_team,
                season,
            )
            continue
        res = Result()
        res.fixture = fixture
        res.home_score = int(home_score)
        res.away_score = int(away_score)
        dbsession.add(res)
    dbsession.commit()


def fill_results_from_api(
    gw_start: int, gw_end: int, season: str, dbsession: Session
) -> None:
    fetcher = FPLDataFetcher()
    matches = fetcher.get_fixture_data()
    if get_last_finished_gameweek() == 0:
        logger.info(
            "No complete gameweeks, skipping match result update for %s season",
            season,
        )
        return
    if (
        get_last_complete_gameweek_in_db(season=season, dbsession=dbsession)
        == get_last_finished_gameweek()
    ):
        logger.info("Match results up-to-date, skipping update for %s season", season)
        return
    for m in track(matches, description=f"RESULTS {season}"):
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
            msg = f"Unable to find team with id {home_id}"
            raise ValueError(msg)
        if not away_team:
            msg = f"Unable to find team with id {away_id}"
            raise ValueError(msg)
        home_score = m["team_h_score"]
        away_score = m["team_a_score"]
        f = find_fixture(
            home_team,
            was_home=True,
            other_team=away_team,
            gameweek=gameweek,
            season=season,
            dbsession=dbsession,
        )
        if f is None:
            logger.warning(
                "Unable to find fixture for %s vs %s in %s gameweek %s",
                home_team,
                away_team,
                season,
                gameweek,
            )
            continue
        if f.result is None:
            res = Result()
            add = True
        else:
            res = f.result
            add = False
        res.fixture = f
        res.home_score = int(home_score)
        res.away_score = int(away_score)
        if add:
            dbsession.add(res)
    dbsession.commit()


def make_result_table(
    seasons: list[str] | None = None, dbsession: Session | None = None
) -> None:
    """
    past seasons - read results from csv
    """
    dbsession = dbsession if dbsession is not None else get_session()
    if seasons is None:
        seasons = []
    if not seasons:
        seasons = [CURRENT_SEASON]
        seasons += get_past_seasons(3)
    for season in sort_seasons(seasons):
        if season == CURRENT_SEASON:
            # current season - use API
            gw_end = next_gameweek(fetcher=get_fetcher())
            fill_results_from_api(1, gw_end, CURRENT_SEASON, dbsession)
        else:
            fill_results_from_csv(resource(f"results_{season}.csv"), season, dbsession)
