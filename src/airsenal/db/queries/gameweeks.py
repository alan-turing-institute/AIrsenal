"""Working out which gameweek we are in, from what the database knows."""

from datetime import UTC, date, datetime
from typing import TYPE_CHECKING

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from airsenal.core.caching import cache_ignoring_session
from airsenal.core.dates import parse_date, parse_datetime
from airsenal.core.logging import get_logger
from airsenal.core.season import CURRENT_SEASON
from airsenal.db.models import Fixture
from airsenal.db.session import get_session

if TYPE_CHECKING:
    # Annotation only, and quoted at every use: db must not import the HTTP client.
    # A caller supplies a fetcher when the database has no fixtures to work the
    # gameweek out from.
    from airsenal.remote.fpl_api import FPLDataFetcher

logger = get_logger(__name__)


class NoFixtureDataError(RuntimeError):
    """
    Raised when the next gameweek cannot be determined because the database holds no
    fixtures for the season and no FPL API fetcher was supplied to fall back on.
    """


@cache_ignoring_session(maxsize=8)
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
    fetcher: "FPLDataFetcher | None" = None,
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
    timenow = datetime.now(UTC)
    fixtures = dbsession.scalars(select(Fixture).where(Fixture.season == season)).all()
    earliest_future_gameweek = get_max_gameweek(season, dbsession=dbsession) + 1

    if len(fixtures) > 0:
        for fixture in fixtures:
            if fixture.date is None or fixture.gameweek is None:
                # date could be null if fixture not scheduled
                continue
            fixture_date = parse_datetime(fixture.date).replace(tzinfo=UTC)
            if fixture_date > timenow and fixture.gameweek < earliest_future_gameweek:
                earliest_future_gameweek = fixture.gameweek

        # now make sure we aren't in the middle of a gameweek
        for fixture in fixtures:
            if not fixture.date:
                # date could be null if fixture not scheduled
                continue
            if (
                parse_datetime(fixture.date).replace(tzinfo=UTC) < timenow
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

        for api_fixture in fixture_data:
            if (
                api_fixture["finished"] is False
                and api_fixture["event"]
                and api_fixture["event"] < earliest_future_gameweek
            ):
                earliest_future_gameweek = api_fixture["event"]
        # check whether we're mid-gameweek
        for api_fixture in fixture_data:
            if (
                api_fixture["finished"] is True
                and api_fixture["event"] == earliest_future_gameweek
            ):
                earliest_future_gameweek += 1
                break

    return earliest_future_gameweek


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
        fetcher: "FPLDataFetcher | None",
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
    fetcher: "FPLDataFetcher | None" = None,
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


@cache_ignoring_session(maxsize=365)
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
    end_season_gw = get_max_gameweek(season, dbsession=dbsession) + 1

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


@cache_ignoring_session(maxsize=365)
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


def get_gameweeks_array(
    n_gameweeks: int | None = None,
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
    if gameweek_end is not None and n_gameweeks is not None:
        msg = "Only one of gameweek_end and n_gameweeks should be defined"
        raise RuntimeError(msg)
    if gameweek_start is None and season != CURRENT_SEASON:
        msg = "gameweek_start must be defined if using previous seasons"
        raise RuntimeError(msg)

    # Set defaults for undefined arguments
    if gameweek_start is None:
        gameweek_start = next_gameweek()
    if gameweek_end is None:
        if n_gameweeks is None:
            # How far ahead to look by default is a decision about a run, not
            # about the gameweek table; it lives in pipeline/settings.py. This
            # function does the arithmetic and has to be told the window.
            msg = "Specify how many gameweeks to cover, or which gameweek to stop at"
            raise RuntimeError(msg)
        gameweek_end = gameweek_start + n_gameweeks

    gameweeks = list(range(gameweek_start, gameweek_end))
    max_gameweek = get_max_gameweek(season=season, dbsession=dbsession)
    gameweeks = list(filter(lambda x: x <= max_gameweek, gameweeks))

    if len(gameweeks) == 0:
        msg = "No gameweeks in specified range"
        raise ValueError(msg)
    if max(gameweeks) < gameweek_end - 1:
        logger.warning(
            "Last gameweek set to %s (%s weeks ahead)", max(gameweeks), len(gameweeks)
        )

    return gameweeks
