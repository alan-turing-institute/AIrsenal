"""
Useful commands to query the database.
"""

from collections.abc import Iterable
from datetime import date, datetime, timezone
from functools import lru_cache
from operator import itemgetter
from pickle import dumps, loads
from typing import TypeVar

import dateparser
import pandas as pd
import regex as re
from curl_cffi import requests
from dateutil.parser import isoparse
from sqlalchemy import case, or_, select
from sqlalchemy.orm import selectinload
from sqlalchemy.orm.session import Session

from airsenal.core.output import console, get_logger, table
from airsenal.domain.season import CURRENT_SEASON
from airsenal.fetch.fpl_api import FPLDataFetcher, get_fetcher
from airsenal.framework.schema import (
    Absence,
    Fixture,
    Player,
    PlayerAttributes,
    PlayerMapping,
    PlayerPrediction,
    PlayerScore,
    Team,
    Transaction,
    get_session,
)

logger = get_logger(__name__)


class NoFixtureDataError(RuntimeError):
    """
    Raised when the next gameweek cannot be determined because the database holds no
    fixtures for the season and no FPL API fetcher was supplied to fall back on.
    """


@lru_cache(1)
def get_max_gameweek(
    season: str = CURRENT_SEASON, dbsession: Session | None = None
) -> int:
    """
    Return the maximum gameweek number across all scheduled fixtures. This should
    generally be 38, but may be different with major disruptions (e.g. Covid-19).
    """
    dbsession = dbsession if dbsession is not None else get_session()
    max_gw_fixture = dbsession.scalars(
        select(Fixture)
        .where(Fixture.season == season, Fixture.gameweek.is_not(None))
        .order_by(Fixture.gameweek.desc())
        .limit(1)
    ).first()
    return (
        38
        if max_gw_fixture is None or max_gw_fixture.gameweek is None
        else max_gw_fixture.gameweek
    )


def get_next_gameweek(
    season: str = CURRENT_SEASON,
    dbsession: Session | None = None,
    *,
    fetcher: FPLDataFetcher | None = None,
) -> int:
    """
    Use the current time to figure out which gameweek we are currently in.

    Recomputed on every call. Prefer `next_gameweek`, which caches the result for the
    lifetime of the process.

    Parameters
    ==========
    season: str
    dbsession: Session or None
    fetcher: FPLDataFetcher or None
        Only consulted when the database holds no fixtures for the season, which
        happens when the database has not been populated yet. If it is None in that
        situation, NoFixtureDataError is raised rather than an HTTP request made.

    Raises
    ======
    NoFixtureDataError
        The database has no fixtures for the season and no fetcher was given.
    """
    dbsession = dbsession if dbsession is not None else get_session()
    timenow = datetime.now(timezone.utc)
    fixtures = dbsession.scalars(select(Fixture).where(Fixture.season == season)).all()
    earliest_future_gameweek = get_max_gameweek(season, dbsession) + 1

    if len(fixtures) > 0:
        for fixture in fixtures:
            if fixture.date is None or fixture.gameweek is None:
                # date could be null if fixture not scheduled
                continue
            fixture_date = parse_datetime(fixture.date).replace(tzinfo=timezone.utc)
            if fixture_date > timenow and fixture.gameweek < earliest_future_gameweek:
                earliest_future_gameweek = fixture.gameweek

        # now make sure we aren't in the middle of a gameweek
        for fixture in fixtures:
            if not fixture.date:
                # date could be null if fixture not scheduled
                continue
            if (
                parse_datetime(fixture.date).replace(tzinfo=timezone.utc) < timenow
                and fixture.gameweek == earliest_future_gameweek
            ):
                earliest_future_gameweek += 1
    else:
        # No fixtures in the database, so we cannot work this out locally. Falling
        # back to the API has to be asked for explicitly: it used to happen
        # implicitly, which meant merely importing this module could make an HTTP
        # request, and made the test suite impossible to run offline.
        if fetcher is None:
            msg = (
                f"No fixtures in the database for {season}, so the next gameweek "
                "cannot be determined. Populate the database with 'airsenal db "
                "create', or pass fetcher to look it up from the FPL API."
            )
            raise NoFixtureDataError(msg)
        fixture_data = fetcher.get_fixture_data()

        if len(fixture_data) == 0:
            # if no fixtures scheduled assume this is start of season before
            # fixtures have been announced
            return 1

        for fixture in fixture_data:
            if (
                fixture["finished"] is False
                and fixture["event"]
                and fixture["event"] < earliest_future_gameweek
            ):
                earliest_future_gameweek = fixture["event"]
        # check whether we're mid-gameweek
        for fixture in fixture_data:
            if (
                fixture["finished"] is True
                and fixture["event"] == earliest_future_gameweek
            ):
                earliest_future_gameweek += 1
                break

    return earliest_future_gameweek


@lru_cache(365)
def parse_datetime(check_date: datetime | str) -> datetime:
    if isinstance(check_date, datetime):
        return check_date
    try:
        dt: datetime | None = isoparse(check_date)
    except (ValueError, TypeError):
        dt = dateparser.parse(check_date)
    if dt is None:
        msg = f"Unable to parse date: {check_date}"
        raise ValueError(msg)
    return dt


@lru_cache(365)
def parse_date(check_date: date | datetime | str) -> date:
    return (
        check_date
        if isinstance(check_date, date)
        else parse_datetime(check_date).date()
    )


@lru_cache(365)
def get_return_gameweek_by_date(
    return_date: date | datetime | str,
    team: str,
    season: str = CURRENT_SEASON,
    dbsession: Session | None = None,
) -> int:
    """
    Use a date, or easily parse-able date string, and team name to determine the
    gameweek of the next match for that team on or after that date. If no match
    is found, return a placeholder gameweek after the end of the season.
    """
    dbsession = dbsession if dbsession is not None else get_session()
    return_date = parse_date(return_date)

    fixtures = dbsession.scalars(
        select(Fixture)
        .where(
            Fixture.season == season,
            or_(Fixture.away_team == team, Fixture.home_team == team),
            Fixture.date.is_not(None),
        )
        .order_by(Fixture.date)
    ).all()

    # default return if no fixture found after the date
    end_season_gw = get_max_gameweek(season, dbsession) + 1

    if len(fixtures) == 0:
        return end_season_gw

    for fixture in fixtures:
        if fixture.date is None or fixture.gameweek is None:
            # should be filtered out by query, but to keep mypy happy
            continue
        fixture_date = parse_date(fixture.date)
        if fixture_date >= return_date:
            return fixture.gameweek

    return end_season_gw


def get_gameweeks_array(
    weeks_ahead: int | None = None,
    gameweek_start: int | None = None,
    gameweek_end: int | None = None,
    season: str = CURRENT_SEASON,
    dbsession: Session | None = None,
) -> list[int]:
    """
    Returns the array containing only the valid (< max_gameweek) game-weeks
    or raise an exception if no game-weeks remaining.
    """
    # Check arguments are valid
    dbsession = dbsession if dbsession is not None else get_session()
    if gameweek_end is not None and weeks_ahead is not None:
        msg = "Only one of gameweek_end and weeks_ahead should be defined"
        raise RuntimeError(msg)
    if gameweek_start is None and season != CURRENT_SEASON:
        msg = "gameweek_start must be defined if using previous seasons"
        raise RuntimeError(msg)

    # Set defaults for undefined arguments
    if weeks_ahead is None:
        weeks_ahead = 3
    if gameweek_start is None:
        gameweek_start = next_gameweek()
    if gameweek_end is None:
        gameweek_end = gameweek_start + weeks_ahead

    gw_range = list(range(gameweek_start, gameweek_end))
    max_gameweek = get_max_gameweek(season=season, dbsession=dbsession)
    gw_range = list(filter(lambda x: x <= max_gameweek, gw_range))

    if len(gw_range) == 0:
        msg = "No gameweeks in specified range"
        raise ValueError(msg)
    if max(gw_range) < gameweek_end - 1:
        logger.warning(
            "Last gameweek set to %s (%s weeks ahead)", max(gw_range), len(gw_range)
        )

    return gw_range


class _GameweekCache:
    """
    Caches the next gameweek per season for the lifetime of the process.

    This replaces the NEXT_GAMEWEEK module constant. The constant was evaluated at
    import, which meant importing utils ran a database query and, on an empty
    database, an FPL API call. Computing it lazily keeps the value stable within a
    run - exactly as a module constant was - while costing nothing until something
    actually asks for it.

    Stability matters for more than speed: the transfer optimiser reads the next
    gameweek inside its search, and a value that changed mid-run (across a deadline,
    say) would make earlier and later decisions disagree.
    """

    def __init__(self) -> None:
        self._by_season: dict[str, int] = {}

    def get(
        self,
        season: str,
        dbsession: Session | None,
        fetcher: FPLDataFetcher | None,
    ) -> int:
        if season not in self._by_season:
            self._by_season[season] = get_next_gameweek(
                season, dbsession, fetcher=fetcher
            )
        return self._by_season[season]

    def set(self, season: str, gameweek: int) -> None:
        self._by_season[season] = gameweek

    def reset(self) -> None:
        self._by_season.clear()


_gameweek_cache = _GameweekCache()


def next_gameweek(
    season: str = CURRENT_SEASON,
    dbsession: Session | None = None,
    *,
    fetcher: FPLDataFetcher | None = None,
) -> int:
    """
    The next gameweek of a season, computed once per process.

    Replaces the former NEXT_GAMEWEEK module constant. See `get_next_gameweek` for the
    uncached computation and for when `fetcher` is needed.
    """
    return _gameweek_cache.get(season, dbsession, fetcher)


def set_next_gameweek(gameweek: int, season: str = CURRENT_SEASON) -> None:
    """
    Pin the next gameweek for a season, overriding whatever the database says.

    Useful when replaying a historical season, and in tests, where the database
    deliberately has no fixtures to derive it from.
    """
    _gameweek_cache.set(season, gameweek)


def reset_gameweek_cache() -> None:
    """
    Forget the cached next gameweek.

    Needed by tests, which swap the database underneath the cache, and by any
    long-running process that has just populated or updated fixtures.
    """
    _gameweek_cache.reset()


def get_next_season(season: str) -> str:
    """
    Convert string e.g. '1819' into one for next season, i.e. '1920'.
    """
    start_year = int(season[:2])
    end_year = int(season[2:])
    next_start_year = (
        f"0{start_year + 1}" if start_year + 1 < 10 else str(start_year + 1)
    )
    next_end_year = f"0{end_year + 1}" if end_year + 1 < 10 else str(end_year + 1)
    return f"{next_start_year}{next_end_year}"


def get_start_end_dates_of_season(season: str) -> list[pd.Timestamp]:
    """
    Obtains rough start and end dates for the season.
    Takes into account the shorter and longer seasons in 19/20 and 20/21.
    """
    start_year = int(f"20{season[:2]}")
    end_year = int(f"20{season[2:]}")
    if season == "1920":
        # regular start, late end to season
        return [pd.Timestamp(2019, 7, 1), pd.Timestamp(2020, 7, 31)]
    if season == "2021":
        # late start to season, regular end
        return [pd.Timestamp(2020, 8, 1), pd.Timestamp(2021, 6, 30)]
    # regular season
    return [pd.Timestamp(start_year, 7, 1), pd.Timestamp(end_year, 6, 30)]


def get_previous_season(season: str) -> str:
    """
    Convert string e.g. '1819' into one for previous season, i.e. '1718'
    """
    start_year = int(season[:2])
    end_year = int(season[2:])
    prev_start_year = start_year - 1
    prev_end_year = end_year - 1
    return f"{prev_start_year}{prev_end_year}"


def get_past_seasons(num_seasons: int) -> list[str]:
    """
    Go back num_seasons from the current one.
    """
    season = CURRENT_SEASON
    seasons = []
    for _ in range(num_seasons):
        season = get_previous_season(season)
        seasons.append(season)
    return seasons


def get_current_players(
    gameweek: int | None = None,
    season: str | None = None,
    fpl_team_id: int | None = None,
    dbsession: Session | None = None,
) -> list[int]:
    """
    Use the transactions table to find the team as of specified gameweek,
    then add up the values at that gameweek using the FPL API data.
    If gameweek is None, get team for next gameweek.
    """
    if not fpl_team_id:
        fpl_team_id = get_fetcher().FPL_TEAM_ID
    if not season:
        season = CURRENT_SEASON
    dbsession = dbsession if dbsession is not None else get_session()
    current_players = []
    transactions = dbsession.scalars(
        select(Transaction)
        .where(
            Transaction.fpl_team_id == fpl_team_id,
            Transaction.free_hit
            == 0,  # free_hit players shouldn't be considered part of squad
            Transaction.season == season,
        )
        .order_by(Transaction.gameweek, Transaction.id)
    ).all()

    if len(transactions) == 0:
        # not updated the transactions table yet
        return []
    for t in transactions:
        if gameweek and t.gameweek > gameweek:
            break
        if t.bought_or_sold == 1:
            current_players.append(t.player_id)
        else:
            current_players.remove(t.player_id)
    assert len(current_players) == 15
    return current_players


def get_bank(
    fpl_team_id: int | None = None,
    gameweek: int | None = None,
    season: str = CURRENT_SEASON,
    fetcher: FPLDataFetcher | None = None,
) -> float:
    """
    Find out how much this FPL team had in the bank before the specified gameweek.
    If gameweek is not provided, give the most recent value.
    If fpl_team_id is not specified, will use the FPL_TEAM_ID environment var, or
    the contents of the file airsenal/data/FPL_TEAM_ID.
    """
    fetcher = fetcher if fetcher is not None else get_fetcher()
    if season != CURRENT_SEASON:
        msg = "Calculating the bank for past seasons not yet implemented"
        raise RuntimeError(msg)

    if not fpl_team_id:
        fpl_team_id = get_fetcher().FPL_TEAM_ID
    # check if we're logged in, which will let us get the most up-to-date info
    try:
        return fetcher.get_current_bank(fpl_team_id)
    except requests.exceptions.RequestException:
        logger.warning(
            "Failed to get actual bank from a logged in API. "
            "Will try to estimate it from the API without logging in, which will "
            "not include any transfers made in the current gameweek.",
            exc_info=True,
        )
        data = fetcher.get_fpl_team_history_data(fpl_team_id)
        if "current" not in data or len(data["current"]) <= 0:
            return 0

        if gameweek and isinstance(gameweek, int):
            for gw in data["current"]:
                if gw["event"] == gameweek - 1:  # value after previous gameweek
                    return gw["bank"]
        # otherwise, return the most recent value
        return data["current"][-1]["bank"]


def get_entry_start_gameweek(
    fpl_team_id: int, fetcher: FPLDataFetcher | None = None
) -> int:
    """
    Find the gameweek an FPL team ID was entered in by searching for the first gameweek
    the API has 'picks' for.
    """
    fetcher = fetcher if fetcher is not None else get_fetcher()
    starting_gw = 1
    while starting_gw < next_gameweek():
        try:
            if get_players_for_gameweek(starting_gw, fpl_team_id, fetcher=fetcher):
                return starting_gw
            starting_gw += 1
        except requests.exceptions.HTTPError:
            starting_gw += 1
        except requests.exceptions.ConnectionError:
            logger.warning(
                "Failed to connect to the API. Assuming team %s"
                " was entered in GW1 which may be incorrect.",
                fpl_team_id,
                exc_info=True,
            )
            return 1

    # if we failed to find picks in any gameweek, or we're before the start of the
    # season, assume this team ID was entered in NEXT_GAMEWEEK
    return next_gameweek()


def get_free_transfers(
    fpl_team_id: int | None = None,
    gameweek: int | None = None,
    season: str = CURRENT_SEASON,
    dbsession: Session | None = None,
    fetcher: FPLDataFetcher | None = None,
    is_replay: bool = False,
) -> int:
    """
    Work out how many free transfers FPL team should have before specified gameweek.
    If gameweek is not provided, give the most recent value.
    If fpl_team_id is not specified, will use the FPL_TEAM_ID environment var, or
    the contents of the file airsenal/data/FPL_TEAM_ID.
    """
    fetcher = fetcher if fetcher is not None else get_fetcher()
    dbsession = dbsession if dbsession is not None else get_session()
    if season == CURRENT_SEASON and not is_replay:
        # we will use the API to estimate num transfers
        resolved_fpl_team_id = (
            fpl_team_id if fpl_team_id is not None else fetcher.FPL_TEAM_ID
        )
        if resolved_fpl_team_id is None:
            msg = "FPL team ID is required to estimate free transfers from the API"
            raise RuntimeError(msg)

        # try to get the most up-to-date info from logged in api
        try:
            return fetcher.get_num_free_transfers(resolved_fpl_team_id)
        except requests.exceptions.RequestException:
            logger.warning(
                "Failed to get actual free transfers from a logged in API. "
                "Will try to estimate it from the API without logging in, which will "
                "not include any transfers used in the current gameweek.",
                exc_info=True,
            )
        # try to calculate free transfers based on previous transfer history in API
        try:
            data = fetcher.get_fpl_team_history_data(resolved_fpl_team_id)
            num_free_transfers = 1
            if "current" in data and len(data["current"]) > 0:
                starting_gw = get_entry_start_gameweek(
                    resolved_fpl_team_id, fetcher=fetcher
                )
                for gw in data["current"]:
                    if gw["event"] <= starting_gw:
                        continue
                    if gw["event_transfers"] == 0 and num_free_transfers < 2:
                        num_free_transfers += 1
                    elif gw["event_transfers"] >= 2:
                        num_free_transfers = 1
                    # if gameweek was specified, and we reached the previous one,
                    # break out of loop.
                    if gameweek and gw["event"] == gameweek - 1:
                        break
            return num_free_transfers
        except requests.exceptions.RequestException:
            logger.warning(
                "Failed to estimate free transfers from the API. "
                "Will estimate from the DB instead, which may be out of date.",
                exc_info=True,
            )

    # historical/simulated data or API failed - fetch from database
    transactions = dbsession.scalars(
        select(Transaction)
        .where(Transaction.fpl_team_id == fpl_team_id, Transaction.bought_or_sold == 1)
        .order_by(Transaction.gameweek, Transaction.id)
    ).all()
    if len(transactions) == 0:
        return 1
    starting_gw = transactions[0].gameweek
    gw_transactions = {}
    for t in transactions:
        if t.gameweek not in gw_transactions:
            gw_transactions[t.gameweek] = 0
        gw_transactions[t.gameweek] += 1
    num_free_transfers = 1
    if gameweek is None and (season != CURRENT_SEASON or is_replay):
        msg = "Gameweek must be specified for historical data"
        raise ValueError(msg)
    gameweek = gameweek or next_gameweek()
    for prev_gw in range(starting_gw + 1, gameweek):
        if prev_gw not in gw_transactions:
            num_free_transfers = 2
        elif gw_transactions[prev_gw] >= 2:
            num_free_transfers = 1

    return num_free_transfers


@lru_cache(maxsize=365)
def get_gameweek_by_date(
    check_date: date | datetime,
    season: str = CURRENT_SEASON,
    dbsession: Session | None = None,
) -> int | None:
    """
    Gameweek of the next fixture on or after the specified date.
    """
    # convert date to a datetime object if it isn't already one.
    dbsession = dbsession if dbsession is not None else get_session()
    check_date = parse_date(check_date)

    fixtures = dbsession.scalars(
        select(Fixture)
        .where(Fixture.season == season, Fixture.date.is_not(None))
        .order_by(Fixture.date)
    ).all()

    for fixture in fixtures:
        if not fixture.date:
            # to keep mypy happy
            continue
        fixture_date = parse_date(fixture.date)
        if fixture_date >= check_date:
            return fixture.gameweek
    return None


def get_team_name(
    team_id: int, season: str = CURRENT_SEASON, dbsession: Session | None = None
) -> str | None:
    """
    Return 3-letter team name given a numerical id.
    These ids are based on alphabetical order of all teams in that season,
    so can vary from season to season.
    """
    dbsession = dbsession if dbsession is not None else get_session()
    team = dbsession.scalars(
        select(Team).where(Team.season == season, Team.team_id == team_id).limit(1)
    ).first()
    if team:
        return team.name
    logger.warning("Unknown team_id %s for %s season", team_id, season)
    return None


def get_player(
    player_name_or_id: str | int,
    dbsession: Session | None = None,
) -> Player | None:
    """
    Query the player table by name, id, or opta_code, and return the player object
    (or None).

    NOTE the player_id that can be passed as an argument here is NOT
    guaranteed to be the id for that player in the FPL API. The one here
    is the entry (primary key) in our database.
    Use the function get_player_from_api_id() to find the player corresponding
    to the FPL API ID.
    """
    dbsession = dbsession if dbsession is not None else get_session()
    # ID field match
    if isinstance(player_name_or_id, str) and player_name_or_id.isdigit():
        player_name_or_id = int(player_name_or_id)

    if isinstance(player_name_or_id, int):
        if p := dbsession.scalars(
            select(Player).where(Player.player_id == player_name_or_id).limit(1)
        ).first():
            return p
        # failed to find player by ID
        return None

    # Name or Opta code match
    if p := dbsession.scalars(
        select(Player)
        .where(
            or_(
                Player.name == player_name_or_id,
                Player.opta_code == player_name_or_id,
            )
        )
        .limit(1)
    ).first():
        return p

    # Alternative name match
    if mapping := dbsession.scalars(
        select(PlayerMapping)
        .where(PlayerMapping.alt_name == player_name_or_id)
        .limit(1)
    ).first():
        return dbsession.scalars(
            select(Player).where(Player.player_id == mapping.player_id).limit(1)
        ).first()

    if p := dbsession.scalars(
        select(Player).where(Player.display_name == player_name_or_id).limit(1)
    ).first():
        return p

    # No match found
    return None


def get_player_from_api_id(
    api_id: int, dbsession: Session | None = None
) -> Player | None:
    """
    Query the database and return the player with corresponding attribute fpl_api_id.
    """
    dbsession = dbsession if dbsession is not None else get_session()
    if p := dbsession.scalars(
        select(Player).where(Player.fpl_api_id == api_id).limit(1)
    ).first():
        return p
    logger.warning("Unable to find player with fpl_api_id %s", api_id)
    return None


def get_player_name(player_id: int, dbsession: Session | None = None) -> str | None:
    """
    Lookup player name, for human readability.
    """
    if p := get_player(player_id, dbsession):
        return str(p)
    logger.warning("Unknown player_id %s", player_id)
    return None


def get_player_id(player_name: str, dbsession: Session | None = None) -> int | None:
    if p := get_player(player_name, dbsession):
        return p.player_id
    logger.warning("Unknown player_name %s", player_name)
    return None


def list_teams(
    season: str = CURRENT_SEASON, dbsession: Session | None = None
) -> list[dict[str, str]]:
    """
    Print all teams from current season.
    """
    dbsession = dbsession if dbsession is not None else get_session()
    rows = dbsession.scalars(select(Team).where(Team.season == season)).all()
    return [{"name": row.name, "full_name": row.full_name} for row in rows]


def list_players(
    position: str = "all",
    team: str = "all",
    order_by: str = "price",
    season: str = CURRENT_SEASON,
    gameweek: int | None = None,
    dbsession: Session | None = None,
) -> list[Player]:
    """
    Print list of players and return a list of player_ids.
    """
    gameweek = next_gameweek() if gameweek is None else gameweek
    dbsession = dbsession if dbsession is not None else get_session()
    # if trying to get players from after DB has filled, return most recent players
    if season == CURRENT_SEASON:
        last_pa = dbsession.scalars(
            select(PlayerAttributes)
            .where(PlayerAttributes.season == season)
            .order_by(PlayerAttributes.gameweek.desc())
            .limit(1)
        ).first()
        if last_pa and gameweek > last_pa.gameweek:
            logger.debug(
                "Incomplete data in DB for GW%s, returning players from GW%s.",
                gameweek,
                last_pa.gameweek,
            )
            gameweek = last_pa.gameweek

    gameweeks = [gameweek]
    # check if the team (or all teams) play in the specified gameweek, if not
    # attributes might be missing
    fixtures = get_fixture_teams(
        get_fixtures_for_gameweek(gameweek, season=season, dbsession=dbsession)
    )
    teams_with_fixture = {t for fixture in fixtures for t in fixture}

    if (team == "all" and len(teams_with_fixture) < 20) or (
        team != "all" and team not in teams_with_fixture
    ):
        # check neighbouring gameweeks to get all 20 teams/specified team
        gws_to_try = [gameweek - 1, gameweek + 1, gameweek - 2, gameweek + 2]
        max_gw = get_max_gameweek(season, dbsession)
        gws_to_try = [gw for gw in gws_to_try if gw > 0 and gw <= max_gw]

        for gw in gws_to_try:
            fixtures = get_fixture_teams(
                get_fixtures_for_gameweek(gw, season=season, dbsession=dbsession)
            )
            new_teams = [t for fixture in fixtures for t in fixture]

            if team == "all" and any(t not in teams_with_fixture for t in new_teams):
                # this gameweek has some teams we haven't seen before
                gameweeks.append(gw)
                for t in new_teams:
                    teams_with_fixture.add(t)
                if len(teams_with_fixture) == 20:
                    break

            elif team != "all" and team in new_teams:
                # this gameweek has the team we're looking for
                gameweeks.append(gw)
                break

    query = select(PlayerAttributes).where(
        PlayerAttributes.season == season,
        PlayerAttributes.gameweek.in_(gameweeks),
    )
    if team != "all":
        query = query.where(PlayerAttributes.team == team)
    if position != "all":
        query = query.where(PlayerAttributes.position == position)
    else:
        # exclude managers
        query = query.where(PlayerAttributes.position != "MNG")
    if len(gameweeks) > 1:
        # Sort query results by order of gameweeks - i.e. make sure the input
        # query gameweek comes first.
        _whens = {gw: i for i, gw in enumerate(gameweeks)}
        sort_order = case(_whens, value=PlayerAttributes.gameweek)
        query = query.order_by(sort_order)
    if order_by == "price":
        query = query.order_by(PlayerAttributes.price.desc())
    players = []
    prices = []
    seen_player_ids = set()
    for pa in dbsession.scalars(query.options(selectinload(PlayerAttributes.player))):
        # might have queried multiple gameweeks with same player returned
        # multiple times - only add if it's a new player
        if pa.player_id in seen_player_ids:
            continue
        seen_player_ids.add(pa.player_id)
        players.append(pa.player)
        prices.append(pa.price)
        if len(gameweeks) == 1 or order_by != "price":
            logger.debug("%s %s %s %s", pa.player, pa.team, pa.position, pa.price)
    if len(gameweeks) > 1 and order_by == "price":
        # Query sorted by gameweek first, so need to do a final sort here to
        # get final price order if more than one gameweek queried.
        sort_players = sorted(
            zip(prices, players, strict=False), reverse=True, key=lambda p: p[0]
        )
        for price, player in sort_players:
            logger.debug("%s %s", player, price)
        players = [p for _, p in sort_players]
    return players


def is_future_gameweek(
    season: str,
    gameweek: int | None,
    current_season: str = CURRENT_SEASON,
    next_gameweek: int | None = None,
) -> bool:
    """
    Return True is season and gameweek refers to a gameweek that is after
    (or the same) as current_season and next_gameweek.
    """
    if next_gameweek is None:
        # The parameter shadows the module-level next_gameweek() function, so go via
        # the cache it reads. Renaming the parameter is not an option: callers pass it
        # by keyword (e.g. prediction_utils.py:754).
        next_gameweek = _gameweek_cache.get(current_season, None, None)
    return (
        season == current_season and (gameweek is None or gameweek >= next_gameweek)
    ) or (season != current_season and int(season) > int(current_season))


def get_max_matches_per_player(
    position: str = "all",
    season: str = CURRENT_SEASON,
    gameweek: int | None = None,
    dbsession: Session | None = None,
) -> int:
    """
    Can be used e.g. in bpl_interface.get_player_history_df
    to help avoid a ragged dataframe.
    """
    gameweek = next_gameweek() if gameweek is None else gameweek
    dbsession = dbsession if dbsession is not None else get_session()
    players = list_players(
        position=position, season=season, gameweek=gameweek, dbsession=dbsession
    )
    player_ids = [p.player_id for p in players if p.player_id is not None]
    if not player_ids:
        return 0

    scores = dbsession.scalars(
        select(PlayerScore)
        .options(selectinload(PlayerScore.fixture))
        .where(PlayerScore.player_id.in_(player_ids))
    ).all()

    matches_per_player = dict.fromkeys(player_ids, 0)
    for score in scores:
        if score.fixture is None or score.player_id is None:
            continue
        if not is_future_gameweek(
            score.fixture.season,
            score.fixture.gameweek,
            current_season=season,
            next_gameweek=gameweek,
        ):
            matches_per_player[score.player_id] += 1

    return max(matches_per_player.values(), default=0)


def get_player_attributes(
    player_name_or_id: str | int,
    season: str = CURRENT_SEASON,
    gameweek: int | None = None,
    dbsession: Session | None = None,
) -> PlayerAttributes | None:
    """
    Get a player's attributes for a given gameweek in a given season.
    """
    gameweek = next_gameweek() if gameweek is None else gameweek
    dbsession = dbsession if dbsession is not None else get_session()
    if isinstance(player_name_or_id, str) and player_name_or_id.isdigit():
        player_id = int(player_name_or_id)
    elif isinstance(player_name_or_id, int):
        player_id = player_name_or_id
    elif isinstance(player_name_or_id, str):
        player = get_player(player_name_or_id)
        if player:
            player_id = player.player_id
        else:
            return None
    return dbsession.scalars(
        select(PlayerAttributes)
        .where(
            PlayerAttributes.season == season,
            PlayerAttributes.gameweek == gameweek,
            PlayerAttributes.player_id == player_id,
        )
        .limit(1)
    ).first()


def get_fixtures_for_player(
    player: Player | str | int,
    season: str = CURRENT_SEASON,
    gw_range: list[int] | None = None,
    dbsession: Session | None = None,
) -> list[Fixture]:
    """
    Search for upcoming fixtures for a player, specified either by id or name.
    If gw_range not specified:
       for current season: return fixtures from now to end of season
       for past seasons: return all fixtures in the season
    """
    dbsession = dbsession if dbsession is not None else get_session()
    if isinstance(player, str):  # given a player name
        player_record = dbsession.scalars(
            select(Player).where(Player.name == player).limit(1)
        ).first()
    elif isinstance(player, int):  # given a player id
        player_record = dbsession.scalars(
            select(Player).where(Player.player_id == player).limit(1)
        ).first()
    else:  # given a player object
        player_record = player
    if not player_record:
        logger.warning("Couldn't find %s in database", player)
        return []
    if not gw_range and season != CURRENT_SEASON:
        msg = "Gameweek range must be specified for past seasons"
        raise ValueError(msg)
    if not gw_range:
        team = player_record.team(season, next_gameweek())
    else:
        team = player_record.team(season, gw_range[0])  # same team for whole gw_range
    tag = get_latest_fixture_tag(season, dbsession)
    fixture_rows = dbsession.scalars(
        select(Fixture)
        .where(
            Fixture.season == season,
            Fixture.tag == tag,
            or_(Fixture.home_team == team, Fixture.away_team == team),
        )
        .order_by(Fixture.gameweek)
    ).all()
    fixtures = []
    for fixture in fixture_rows:
        if not fixture.gameweek:  # fixture not scheduled yet
            continue
        if gw_range:
            if fixture.gameweek in gw_range:
                fixtures.append(fixture)
        else:
            if season == CURRENT_SEASON and fixture.gameweek < next_gameweek():
                continue
            logger.debug("%s", fixture)
            fixtures.append(fixture)
    return fixtures


def get_next_fixture_for_player(
    player: Player | str | int,
    season: str = CURRENT_SEASON,
    gameweek: int | None = None,
    dbsession: Session | None = None,
) -> str:
    """
    Get a players next fixture as a string, for easy displaying.
    """
    gameweek = next_gameweek() if gameweek is None else gameweek
    dbsession = dbsession if dbsession is not None else get_session()
    # given a player name or id, convert to player object
    if isinstance(player, str | int):
        maybe_player = get_player(player, dbsession)
        if not maybe_player:
            logger.warning("Couldn't find player %s in database", player)
            return ""
        player = maybe_player
    team = player.team(season, gameweek)
    fixtures_for_player = get_fixtures_for_player(player, season, [gameweek], dbsession)
    output_string = ""
    for fixture in fixtures_for_player:
        if fixture.home_team == team:
            output_string += fixture.away_team + " (h)"
        else:
            output_string += fixture.home_team + " (a)"
        output_string += ", "
    return output_string[:-2]


def get_fixtures_for_season(
    season: str = CURRENT_SEASON, dbsession: Session | None = None
) -> list[Fixture]:
    """
    Return all fixtures for a season.
    """
    dbsession = dbsession if dbsession is not None else get_session()
    return list(
        dbsession.scalars(select(Fixture).where(Fixture.season == season)).all()
    )


def get_fixtures_for_gameweek(
    gameweek: list[int] | int,
    season: str = CURRENT_SEASON,
    dbsession: Session | None = None,
) -> list[Fixture]:
    """
    Get a list of fixtures for the specified gameweek(s).
    """
    dbsession = dbsession if dbsession is not None else get_session()
    if isinstance(gameweek, int):
        gameweek = [gameweek]
    return list(
        dbsession.scalars(
            select(Fixture).where(
                Fixture.season == season, Fixture.gameweek.in_(gameweek)
            )
        ).all()
    )


def get_fixture_teams(fixtures: Iterable[Fixture]) -> list[tuple[str, str]]:
    """
    Get (home_team, away_team) tuples for each fixture in a list of fixtures.
    """
    return [(fixture.home_team, fixture.away_team) for fixture in fixtures]


def get_player_scores(
    fixture: Fixture | None = None,
    player: Player | None = None,
    dbsession: Session | None = None,
) -> list[PlayerScore] | PlayerScore | None:
    """
    Get player scores for a fixture.
    """
    dbsession = dbsession if dbsession is not None else get_session()
    if fixture is None and player is None:
        msg = "At least one of fixture and player must be defined"
        raise ValueError(msg)

    query = select(PlayerScore)
    if fixture is not None:
        query = query.where(PlayerScore.fixture_id == fixture.fixture_id)
    if player is not None:
        query = query.where(PlayerScore.player_id == player.player_id)

    player_scores = list(dbsession.scalars(query).all())
    if not player_scores:
        return None

    if fixture is not None and player is not None:
        if len(player_scores) > 1:
            msg = f"More than one score found for player {player} in fixture {fixture}"
            raise ValueError(msg)
        return player_scores[0]
    return player_scores


def get_players_for_gameweek(
    gameweek: int,
    fpl_team_id: int | None = None,
    fetcher: FPLDataFetcher | None = None,
) -> list[Player]:
    """
    Use FPL API to get the players for a given gameweek.
    """
    fetcher = fetcher if fetcher is not None else get_fetcher()
    if not fpl_team_id:
        fpl_team_id = get_fetcher().FPL_TEAM_ID

    player_data = fetcher.get_fpl_team_data(gameweek, fpl_team_id)["picks"]
    player_api_id_list = [p["element"] for p in player_data]
    players: list[Player] = []
    for api_id in player_api_id_list:
        player = get_player_from_api_id(api_id)
        if player is None:
            logger.warning("Unable to find player with fpl_api_id %s", api_id)
            continue
        players.append(player)
    return players


def get_previous_points_for_same_fixture(
    player: str | int, fixture_id: int, dbsession: Session | None = None
) -> dict[str, int]:
    """
    Search the past matches for same fixture in past seasons,
    and how many points the player got.
    """
    dbsession = dbsession if dbsession is not None else get_session()
    if isinstance(player, str):
        player_record = dbsession.scalars(
            select(Player).where(Player.name == player).limit(1)
        ).first()
        if not player_record:
            logger.warning("Can't find player %s", player)
            return {}
        player_id = player_record.player_id
    else:
        player_id = player
    fixture = dbsession.scalars(
        select(Fixture).where(Fixture.fixture_id == fixture_id).limit(1)
    ).first()
    if not fixture:
        logger.warning("Couldn't find fixture_id %s", fixture_id)
        return {}
    home_team = fixture.home_team
    away_team = fixture.away_team

    previous_matches = dbsession.scalars(
        select(Fixture)
        .where(Fixture.home_team == home_team, Fixture.away_team == away_team)
        .order_by(Fixture.season)
    ).all()
    fixture_seasons = {
        f.fixture_id: f.season for f in previous_matches if f.fixture_id is not None
    }
    if not fixture_seasons:
        return {}

    previous_points = {}
    scores = dbsession.scalars(
        select(PlayerScore).where(
            PlayerScore.player_id == player_id,
            PlayerScore.fixture_id.in_(fixture_seasons.keys()),
        )
    ).all()
    for score in scores:
        if score.fixture_id is None:
            continue
        season = fixture_seasons.get(score.fixture_id)
        if season is not None:
            previous_points[season] = score.points

    return previous_points


@lru_cache(maxsize=4096)
def get_predicted_points_for_player(
    player: Player | str | int,
    tag: str,
    season: str = CURRENT_SEASON,
    dbsession: Session | None = None,
) -> dict[int, float]:
    """
    Query the player prediction table for a given player.
    Return a dict, keyed by gameweek.
    """
    dbsession = dbsession if dbsession is not None else get_session()
    if isinstance(player, str | int):
        maybe_player = get_player(player, dbsession=dbsession)
        if maybe_player is None:
            msg = f"Couldn't find player {player} in database"
            raise ValueError(msg)
        player = maybe_player

    pps = dbsession.scalars(
        select(PlayerPrediction)
        .options(selectinload(PlayerPrediction.fixture))
        .where(
            PlayerPrediction.fixture.has(Fixture.season == season),
            PlayerPrediction.player_id == player.player_id,
            PlayerPrediction.tag == tag,
        )
    ).all()
    ppdict = {}
    for prediction in pps:
        # there is one prediction per fixture.
        # for double gameweeks, we need to add the two together
        gameweek = prediction.fixture.gameweek
        if gameweek is None:
            logger.warning(
                "Player %s has no gameweek for fixture %s", player, prediction.fixture
            )
            continue
        if gameweek not in ppdict:
            ppdict[gameweek] = 0.0
        ppdict[gameweek] += prediction.predicted_points
    # we still need to fill in zero for gameweeks that they're not playing.
    max_gw = get_max_gameweek(season, dbsession)
    for gw in range(1, max_gw + 1):
        if gw not in ppdict:
            ppdict[gw] = 0.0
    return ppdict


def get_predicted_points(
    gameweek: int | list[int],
    tag: str,
    position: str = "all",
    team: str = "all",
    season: str = CURRENT_SEASON,
    dbsession: Session | None = None,
) -> list[tuple[Player, float]]:
    """
    Query the player_prediction table with selections, return
    list of tuples (player_id, predicted_points) ordered by predicted_points
    "gameweek" argument can either be a single integer for one gameweek, or a
    list of gameweeks, in which case we will get the sum over all of them.
    """
    dbsession = dbsession if dbsession is not None else get_session()
    gameweeks = [gameweek] if isinstance(gameweek, int) else gameweek
    players = list_players(
        position,
        team,
        season=season,
        gameweek=gameweeks[0],
        dbsession=dbsession,
    )
    player_ids = [p.player_id for p in players if p.player_id is not None]
    points_by_player = dict.fromkeys(player_ids, 0.0)

    if player_ids:
        rows = dbsession.execute(
            select(
                PlayerPrediction.player_id,
                Fixture.gameweek,
                PlayerPrediction.predicted_points,
            )
            .join(Fixture, PlayerPrediction.fixture_id == Fixture.fixture_id)
            .where(
                PlayerPrediction.player_id.in_(player_ids),
                PlayerPrediction.tag == tag,
                Fixture.season == season,
                Fixture.gameweek.in_(gameweeks),
            )
        ).all()
        for row in rows:
            if row.player_id is not None:
                points_by_player[row.player_id] += row.predicted_points

    output_list = [(p, points_by_player.get(p.player_id, 0.0)) for p in players]
    output_list.sort(key=itemgetter(1), reverse=True)
    return output_list


def get_top_predicted_points(
    gameweek: int | list[int] | None = None,
    tag: str | None = None,
    position: str = "all",
    team: str = "all",
    n_players: int = 10,
    per_position: bool = False,
    max_price: float | None = None,
    season: str = CURRENT_SEASON,
    dbsession: Session | None = None,
) -> None:
    """
    Print players with the top predicted points.

    Keyword Arguments:
        gameweek {int or list} -- Single gameweek or list of gameweeks in which
        case returned totals are sums across all gameweeks (default: next
        gameweek).
        tag {str} -- Prediction tag to query (default: latest prediction tag)
        position {str} -- Player position to query (default: {"all"})
        per_position {boolean} -- If True print top n_players players for
        each position separately (default: {False})
        team {str} -- Team to query (default: {"all"})
        n_players {int} -- Number of players to return (default: {10})
        season {str} -- Season to query (default: {CURRENT_SEASON})
        dbsession {SQLAlchemy session} -- Database session (default: {None})
    """
    dbsession = dbsession if dbsession is not None else get_session()
    discord_webhook = get_fetcher().DISCORD_WEBHOOK
    if not tag:
        tag = get_latest_prediction_tag()
    if not gameweek:
        gameweek = next_gameweek()

    discord_embed = {
        "title": "AIrsenal webhook",
        "description": f"PREDICTED TOP {n_players} PLAYERS FOR GAMEWEEK(S) {gameweek}:",
        "color": 0x35A800,
        "fields": [],
    }

    first_gw = gameweek[0] if isinstance(gameweek, list) else gameweek
    gw_range = (
        f"{first_gw}–{gameweek[-1]}"  # noqa: RUF001
        if isinstance(gameweek, list) and gameweek[-1] != first_gw
        else f"{first_gw}"
    )
    table_title = f"Top {n_players} Predicted Players for Gameweek(s) {gw_range}"

    def print_predictions(predictions: list[tuple[Player, float]], title: str) -> None:
        prediction_table = table(
            "#", "Player", "Team", "Position", "Price", "Predicted Points", title=title
        )
        for rank, (player, predicted_points) in enumerate(predictions[:n_players], 1):
            price = player.price(season, first_gw)
            price_string = f"£{price / 10}m" if price is not None else "Unknown"
            prediction_table.add_row(
                str(rank),
                str(player),
                str(player.team(season, first_gw)),
                str(player.position(season)),
                price_string,
                f"{predicted_points:.2f}",
            )
        console.print(prediction_table)

    if not per_position:
        pts = get_predicted_points(
            gameweek,
            tag,
            position=position,
            team=team,
            season=season,
            dbsession=dbsession,
        )
        if max_price is not None:
            for p in pts:
                price = p[0].price(season, first_gw)
                if price is not None and price > max_price:
                    pts.remove(p)

        pts = sorted(pts, key=lambda x: x[1], reverse=True)

        print_predictions(pts, table_title)

        # If a valid discord webhook URL has been stored
        # in env variables, send a webhook message
        if discord_webhook:
            # Use regex to check the discord webhook url is correctly formatted
            if re.match(
                r"^.*(discord|discordapp)\.com\/api"
                r"\/webhooks\/([\d]+)\/([a-zA-Z0-9_-]+)$",
                discord_webhook,
            ):
                # Maximum fields on a discord embed is 25, so limit this to n_players=8
                payload = predicted_points_discord_payload(
                    discord_embed=discord_embed,
                    position=position,
                    pts=pts[: min(n_players, 8)],
                    season=season,
                    first_gw=first_gw,
                )
                result = requests.post(discord_webhook, json=payload)
                if 200 <= result.status_code < 300:
                    logger.info(
                        "Discord webhook sent, status code: %s", result.status_code
                    )
                else:
                    logger.warning(
                        "Not sent with %s,response:\n{result.json()}",
                        result.status_code,
                    )
            else:
                logger.warning("Discord webhook url is malformed!\n%s", discord_webhook)
    else:
        for i, position in enumerate(["GK", "DEF", "MID", "FWD"]):
            pts = get_predicted_points(
                gameweek,
                tag,
                position=position,
                team=team,
                season=season,
                dbsession=dbsession,
            )
            if max_price is not None:
                for p in pts:
                    maybe_price = p[0].price(season, first_gw)
                    if maybe_price is not None and maybe_price > max_price:
                        pts.remove(p)

            pts = sorted(pts, key=lambda x: x[1], reverse=True)
            title = f"{table_title}\n{position}" if i == 0 else position
            print_predictions(pts, title)

            discord_embed["fields"] = []
            # If a valid discord webhook URL has been stored
            # in env variables, send a webhook message
            if discord_webhook is not None:
                # Use regex to check the discord webhook url is correctly formatted
                if re.match(
                    r"^.*(discord|discordapp)\.com\/api"
                    r"\/webhooks\/([\d]+)\/([a-zA-Z0-9_-]+)$",
                    discord_webhook,
                ):
                    # create a formatted team lineup message for the discord webhook
                    # Maximum fields on a discord embed is 25
                    # limit this to n_players=8
                    payload = predicted_points_discord_payload(
                        discord_embed=discord_embed,
                        position=position,
                        pts=pts[: min(n_players, 8)],
                        season=season,
                        first_gw=first_gw,
                    )
                    result = requests.post(discord_webhook, json=payload)
                    if 200 <= result.status_code < 300:
                        logger.info(
                            "Discord webhook sent, status code: %s", result.status_code
                        )
                    else:
                        logger.warning(
                            "Not sent with %s, response:\n%s",
                            result.status_code,
                            result.json(),
                        )
                else:
                    logger.warning(
                        "Discord webhook url is malformed!\n%s", discord_webhook
                    )


def predicted_points_discord_payload(
    discord_embed: dict,
    position: str,
    pts: list[tuple[Player, float]],
    season: str,
    first_gw: int,
) -> dict:
    """
    json formated discord webhook contentent.
    """
    discord_embed["fields"].append(
        {
            "name": "Position",
            "value": str(position),
            "inline": False,
        }
    )
    for i, p in enumerate(pts):
        price = p[0].price(season, first_gw)
        price_str = str(price / 10) if price is not None else "UNKNOWN_PRICE"
        discord_embed["fields"].extend(
            [
                {
                    "name": "Player",
                    "value": f"{i + 1}. {p[0]}",
                    "inline": True,
                },
                {
                    "name": "Predicted points",
                    "value": f"{p[1]:.2f}pts",
                    "inline": True,
                },
                {
                    "name": "Attributes",
                    "value": (
                        f"£{price_str}m, "
                        f"{p[0].position(season)}, {p[0].team(season, first_gw)}"
                    ),
                    "inline": True,
                },
            ]
        )
    return {
        "content": "",
        "username": "AIrsenal",
        "embeds": [discord_embed],
    }


def get_return_gameweek_from_news(
    news: str, team: str, season: str = CURRENT_SEASON, dbsession: Session | None = None
) -> int | None:
    """
    Parse news strings from the FPL API for the return date of injured or
    suspended players. If a date is found, determine and return the gameweek it
    corresponds to.
    """
    dbsession = dbsession if dbsession is not None else get_session()
    rd_rex = "(Expected back|Suspended until)[\\s]+([\\d]+[\\s][\\w]{3})"
    search_results = re.search(rd_rex, news)
    if not search_results:
        return None

    return_str = search_results.groups()[1]
    # return_str should be a day and month string (without year)

    # create a date in the future from the day and month string
    return_date = dateparser.parse(return_str, settings={"PREFER_DATES_FROM": "future"})
    if not return_date:
        msg = f"Failed to parse date from string '{return_date}'"
        raise ValueError(msg)

    return get_return_gameweek_by_date(
        return_date.date(), team=team, season=season, dbsession=dbsession
    )


def calc_average_minutes(player_scores: list[PlayerScore]) -> float:
    """
    Simple average of minutes played for a list of PlayerScore objects.
    """
    total = 0.0
    for ps in player_scores:
        total += ps.minutes
    return total / len(player_scores)


def estimate_minutes_from_prev_season(
    player: Player,
    season: str = CURRENT_SEASON,
    gameweek: int | None = None,
    n_games_to_use: int = 10,
    exclude_unavailable: bool = True,
    current_team_only: bool = True,
    dbsession: Session | None = None,
) -> list[float]:
    """
    Take average of minutes from previous season if any, or else return [0]
    """
    gameweek = next_gameweek() if gameweek is None else gameweek
    dbsession = dbsession if dbsession is not None else get_session()
    previous_season = get_previous_season(season)

    # Only consider minutes the player played with his current team
    current_team = player.team(season, gameweek)
    query = (
        select(PlayerScore)
        .join(Fixture, PlayerScore.fixture)
        .where(
            PlayerScore.player_id == player.player_id,
            Fixture.season == previous_season,
        )
    )

    if current_team_only:
        current_team = player.team(season, gameweek)
        query = query.where(PlayerScore.player_team == current_team)

    if exclude_unavailable:
        query = query.where(
            or_(
                PlayerScore.minutes >= 60,
                PlayerScore.chance_of_playing == 100,
                PlayerScore.chance_of_playing.is_(None),  # for backwards compatibility
            )
        )

    player_scores = list(
        dbsession.scalars(
            query.order_by(Fixture.gameweek.desc()).limit(n_games_to_use)
        ).all()
    )

    if len(player_scores) == 0:
        # no FPL history / didn't play for current team last season
        return [0]

    # Return average minutes. A weakness of this is increased rotation at the end of the
    # season when teams don't have anything to play for.
    return [calc_average_minutes(player_scores)]


def get_recent_playerscore_rows(
    player: Player,
    num_match_to_use: int = 3,
    season: str = CURRENT_SEASON,
    last_gw: int | None = None,
    exclude_unavailable: bool = False,
    current_team_only: bool = False,
    dbsession: Session | None = None,
) -> list[PlayerScore]:
    """
    Query the playerscore table in the database to retrieve
    the last num_match_to_use rows for this player.
    """
    dbsession = dbsession if dbsession is not None else get_session()
    # If asking for gameweeks without results in DB, revert to most recent results.
    last_available_gameweek = get_last_complete_gameweek_in_db(
        season=season, dbsession=dbsession
    )
    if not last_available_gameweek:
        # e.g. before this season has started
        return []

    if last_gw is None and season != CURRENT_SEASON:
        msg = "last_gw must be specified is running on previous seasons"
        raise ValueError(msg)

    if last_gw is None or last_gw > last_available_gameweek:
        last_gw = last_available_gameweek

    # get the playerscore rows from the db
    query = (
        select(PlayerScore)
        .join(Fixture, PlayerScore.fixture_id == Fixture.fixture_id)
        .where(
            Fixture.season == season,
            PlayerScore.player_id == player.player_id,
            Fixture.gameweek <= last_gw,
        )
    )
    if exclude_unavailable:
        # minutes at least 60 or no flag status (100% chance of playing)
        query = query.where(
            or_(
                PlayerScore.minutes >= 60,
                PlayerScore.chance_of_playing == 100,
                PlayerScore.chance_of_playing.is_(None),  # for backwards compatibility
            )
        )
    if current_team_only:
        team = player.team(season, last_gw)
        query = query.where(PlayerScore.player_team == team)

    return list(
        dbsession.scalars(
            query.order_by(Fixture.gameweek.desc()).limit(num_match_to_use)
        ).all()
    )


def get_playerscores_for_player_gameweek(
    player: Player,
    gameweek: int,
    season: str = CURRENT_SEASON,
    dbsession: Session | None = None,
) -> list[PlayerScore]:
    """
    FPL points for this player for selected match.
    Returns a PlayerScore object.
    """
    dbsession = dbsession if dbsession is not None else get_session()
    return list(
        dbsession.scalars(
            select(PlayerScore)
            .join(Fixture, PlayerScore.fixture_id == Fixture.fixture_id)
            .where(
                Fixture.season == season,
                PlayerScore.player_id == player.player_id,
                Fixture.gameweek == gameweek,
            )
        ).all()
    )


def get_recent_scores_for_player(
    player: Player,
    num_match_to_use: int = 3,
    season: str = CURRENT_SEASON,
    last_gw: int | None = None,
    exclude_unavailable: bool = False,
    current_team_only: bool = False,
    dbsession: Session | None = None,
) -> dict[int, int]:
    """
    Look num_match_to_use matches back, and return the
    FPL points for this player for each of these matches.
    Return a dict {gameweek: score, }
    """
    dbsession = dbsession if dbsession is not None else get_session()
    if last_gw is None:
        if season != CURRENT_SEASON:
            msg = "last_gw must be specified if running on previous seasons"
            raise ValueError(msg)
        last_gw = next_gameweek()
    first_gw = last_gw - num_match_to_use

    playerscores = get_recent_playerscore_rows(
        player,
        num_match_to_use,
        season,
        last_gw,
        exclude_unavailable,
        current_team_only,
        dbsession,
    )
    if not playerscores:  # e.g. start of season
        return {}

    return {range(first_gw, last_gw)[i]: ps.points for i, ps in enumerate(playerscores)}


def get_recent_minutes_for_player(
    player: Player,
    num_match_to_use: int = 3,
    season: str = CURRENT_SEASON,
    last_gw: int | None = None,
    exclude_unavailable: bool = True,
    current_team_only: bool = True,
    dbsession: Session | None = None,
) -> list[float]:
    """
    Look back num_match_to_use matches, and return an array
    containing minutes played in each.
    If current_gw is not given, we take it to be the most
    recent finished gameweek.
    """
    dbsession = dbsession if dbsession is not None else get_session()
    if last_gw is None:
        if season != CURRENT_SEASON:
            msg = "last_gw must be defined if running on previous seasons"
            raise ValueError(msg)
        last_gw = next_gameweek()

    playerscores = (
        get_recent_playerscore_rows(
            player,
            num_match_to_use,
            season,
            last_gw,
            exclude_unavailable,
            current_team_only,
            dbsession,
        )
        or []
    )

    minutes = [float(r.minutes) for r in playerscores]

    if len(minutes) < num_match_to_use:
        minutes += estimate_minutes_from_prev_season(
            player, season, gameweek=last_gw, dbsession=dbsession
        )
    return minutes or [0.0]


def was_historic_absence(
    player: Player, gameweek: int, season: str, dbsession: Session | None = None
) -> bool:
    """
    For past seasons, query the Absence table for a given player and season,
    and see if the gameweek is within the period of the absence.

    Returns: bool, True if player was absent (injured or suspended), False otherwise.
    """
    if season == CURRENT_SEASON:
        # we only consider past seasons here
        return False
    dbsession = dbsession if dbsession is not None else get_session()
    absence = dbsession.scalars(
        select(Absence)
        .where(
            Absence.season == season,
            Absence.player_id == player.player_id,
            Absence.gw_from < gameweek,
            Absence.gw_until > gameweek,
        )
        .limit(1)
    ).first()
    return bool(absence)


def get_last_complete_gameweek_in_db(
    season: str = CURRENT_SEASON, dbsession: Session | None = None
) -> int | None:
    """
    Query the result table to see what was the last gameweek for which
    we have filled the data.
    """
    dbsession = dbsession if dbsession is not None else get_session()
    first_missing = dbsession.scalars(
        select(Fixture)
        .where(
            Fixture.season == season,
            ~Fixture.result.has(),
            Fixture.gameweek.is_not(None),
        )
        .order_by(Fixture.gameweek)
        .limit(1)
    ).first()
    if first_missing is not None and first_missing.gameweek is not None:
        return first_missing.gameweek - 1
    if season == CURRENT_SEASON:
        return None
    return get_max_gameweek(season=season, dbsession=dbsession)


def get_last_finished_gameweek() -> int:
    """
    Query the API to see what the last gameweek marked as 'finished' is.
    """
    event_data = get_fetcher().get_event_data()
    last_finished = 0
    for gw in sorted(event_data.keys()):
        if event_data[gw]["is_finished"]:
            last_finished = gw
        else:
            return last_finished
    return last_finished


def get_latest_prediction_tag(
    season: str = CURRENT_SEASON,
    tag_prefix: str = "",
    dbsession: Session | None = None,
) -> str:
    """
    Query the predicted_score table and get the tag field for the last row.
    """
    dbsession = dbsession if dbsession is not None else get_session()
    query = select(PlayerPrediction).where(
        PlayerPrediction.fixture.has(Fixture.season == season)
    )
    if tag_prefix:
        query = query.where(PlayerPrediction.tag.startswith(tag_prefix))

    latest_prediction = dbsession.scalars(
        query.order_by(PlayerPrediction.id.desc()).limit(1)
    ).first()
    if latest_prediction is None:
        msg = (
            "No predicted points in database - has the database been filled?\n"
            "To calculate points predictions (and fill the database) use "
            "'airsenal_run_prediction'. This should be done before using "
            "'airsenal_make_squad' or 'airsenal_run_optimization'."
        )
        raise RuntimeError(msg)
    return latest_prediction.tag


def get_latest_fixture_tag(
    season: str = CURRENT_SEASON, dbsession: Session | None = None
) -> str:
    """
    Query the predicted_score table and get the tag field for the last row.
    """
    dbsession = dbsession if dbsession is not None else get_session()
    latest_fixture = dbsession.scalars(
        select(Fixture)
        .where(Fixture.season == season)
        .order_by(Fixture.fixture_id.desc())
        .limit(1)
    ).first()
    if latest_fixture is None:
        msg = f"No fixtures found in database for season {season}"
        raise RuntimeError(msg)
    return latest_fixture.tag


def find_fixture(
    team: str | int,
    was_home: bool | None = None,
    other_team: str | int | None = None,
    gameweek: int | None = None,
    season: str = CURRENT_SEASON,
    kickoff_time: date | datetime | str | None = None,
    dbsession: Session | None = None,
    verbose: bool = True,
) -> Fixture | None:
    """
    Get a fixture given a team and optionally whether the team was at home or away,
    the season, kickoff time and the other team in the fixture. Only returns the fixture
    if exactly one match is found, otherwise raises a ValueError.
    """
    dbsession = dbsession if dbsession is not None else get_session()
    if not isinstance(team, str):
        team_name = get_team_name(team, season=season, dbsession=dbsession)
    else:
        team_name = team

    if not team_name:
        msg = f"No team with id {team} in {season} season"
        raise ValueError(msg)

    if isinstance(other_team, int):
        other_team_name = get_team_name(other_team, season=season, dbsession=dbsession)
    else:
        other_team_name = other_team

    query = select(Fixture).where(Fixture.season == season)
    if gameweek:
        query = query.where(Fixture.gameweek == gameweek)
    if was_home is True:
        query = query.where(Fixture.home_team == team_name)
    elif was_home is False:
        query = query.where(Fixture.away_team == team_name)
    else:
        query = query.where(
            or_(Fixture.away_team == team_name, Fixture.home_team == team_name)
        )

    if other_team_name:
        if was_home is True:
            query = query.where(Fixture.away_team == other_team_name)
        elif was_home is False:
            query = query.where(Fixture.home_team == other_team_name)
        elif was_home is None:
            query = query.where(
                or_(
                    Fixture.away_team == other_team_name,
                    Fixture.home_team == other_team_name,
                )
            )

    fixtures = dbsession.scalars(query).all()

    if not fixtures or len(fixtures) == 0:
        if verbose:
            logger.warning(
                "No fixture with season=%s, gw=%s, team_name=%s, was_home=%s, "
                "other_team_name=%s, kickoff_time=%s",
                season,
                gameweek,
                team_name,
                was_home,
                other_team_name,
                kickoff_time,
            )
        return None

    if len(fixtures) == 1:
        return fixtures[0]
    if kickoff_time:
        # team played multiple games in the gameweek, determine the
        # fixture of interest using the kickoff time,
        kickoff_date = parse_date(kickoff_time)

        for f in fixtures:
            f_date = parse_date(f.date)
            if f_date == kickoff_date:
                return f

    logger.warning(
        "No unique fixture with season=%s, gw=%s, team_name=%s, was_home=%s, "
        "kickoff_time=%s",
        season,
        gameweek,
        team_name,
        was_home,
        kickoff_time,
    )
    return None


def get_player_team_from_fixture(
    fixture: Fixture,
    opponent: str | int | None = None,
    player_at_home: bool | None = None,
    season: str = CURRENT_SEASON,
    dbsession: Session | None = None,
) -> str:
    """
    Get the team a player played for given the gameweek, opponent, time and
    whether they were home or away.
    If return_fixture is True, return a tuple of (team_name, fixture).
    """
    dbsession = dbsession if dbsession is not None else get_session()
    if opponent is None and player_at_home is None:
        msg = "Either opponent or player_at_home must be specified"
        raise ValueError(msg)

    if player_at_home is not None:
        return fixture.home_team if player_at_home else fixture.away_team

    if isinstance(opponent, int):
        opponent_name = get_team_name(opponent, season=season, dbsession=dbsession)
    else:
        opponent_name = opponent

    if fixture.home_team == opponent_name:
        return fixture.away_team
    if fixture.away_team == opponent_name:
        return fixture.home_team

    msg = f"Opponent {opponent_name} not in fixture"
    raise ValueError(msg)


def is_transfer_deadline_today() -> bool:
    """
    Return True if there is a transfer deadline later today.
    """
    deadlines = get_fetcher().get_transfer_deadlines()
    for deadline in deadlines:
        deadline_datetime = datetime.strptime(deadline, "%Y-%m-%dT%H:%M:%SZ")
        if (deadline_datetime - datetime.now()).days == 0:
            return True
    return False


T = TypeVar("T")


def fastcopy(obj: T) -> T:
    """
    Faster replacement for copy.deepcopy().
    """
    return loads(dumps(obj, -1))
