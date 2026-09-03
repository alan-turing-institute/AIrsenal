"""Working out which gameweek we are in, from what the database knows."""

from datetime import UTC, date, datetime
from functools import cache
from typing import TYPE_CHECKING

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from airsenal.core.caching import cache_ignoring_session, register_cache
from airsenal.core.dates import parse_date, parse_datetime
from airsenal.core.logging import get_logger
from airsenal.db.models import Fixture
from airsenal.db.session import get_session
from airsenal.game.season import CURRENT_SEASON

if TYPE_CHECKING:
    # Annotation only, and quoted at every use: db must not import the HTTP client.
    # A caller supplies a fetcher when the database has no fixtures to work the
    # gameweek out from.
    from airsenal.remote.fpl_api import FPLDataFetcher

logger = get_logger(__name__)


class NoFixtureDataError(RuntimeError):
    """
    Raised when the next gameweek cannot be determined.

    The database holds no fixtures for the current season, and no FPL API fetcher
    was supplied to fall back on.
    """


@cache_ignoring_session(maxsize=8)
def get_max_gameweek(
    season: str = CURRENT_SEASON, dbsession: Session | None = None
) -> int:
    """
    Return the maximum gameweek number across all scheduled fixtures.

    Generally 38, but may differ after major disruption (e.g. Covid-19). Falls
    back to 38 if the season has no fixtures with a gameweek, so an empty
    database gives a usable answer rather than an error.
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


@cache
def next_gameweek(fetcher: "FPLDataFetcher | None" = None) -> int:
    """
    Use the current time to figure out which gameweek we are currently in.

    Only the current season has a next gameweek: a replay of a past one is told
    which gameweek it is up to. The answer is worked out once and then held for the
    lifetime of the process, so that a run gets one throughout - the transfer
    optimiser reads this inside its search, and a value that changed mid-run, across
    a deadline say, would make earlier and later decisions disagree. Worked out from
    the default database, and dropped by `clear_query_caches` when something writes
    what it reads or points the package at another database.

    Args:
        fetcher: Only consulted when the database holds no fixtures, which happens
            when the database has not been populated yet.

    Raises:
        NoFixtureDataError: The database has no fixtures and no fetcher was given,
            so there is nothing to fall back on.
    """
    dbsession = get_session()
    timenow = datetime.now(UTC)
    fixtures = dbsession.scalars(
        select(Fixture).where(Fixture.season == CURRENT_SEASON)
    ).all()
    earliest_future_gameweek = get_max_gameweek(dbsession=dbsession) + 1

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
        # No fixtures in the database, so we cannot work this out locally.
        # Falling back to the API has to be asked for explicitly, so that nothing
        # makes an HTTP request just by calling this.
        if fetcher is None:
            msg = (
                f"No fixtures in the database for {CURRENT_SEASON}, so the next "
                "gameweek cannot be determined. Populate the database with "
                "'airsenal db create', or pass fetcher to look it up from the "
                "FPL API."
            )
            raise NoFixtureDataError(msg)
        fixture_data = fetcher.get_fixture_data()

        if len(fixture_data) == 0:
            # if no fixtures scheduled assume this is start of season before
            # fixtures have been announced
            earliest_future_gameweek = 1
        else:
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


register_cache(next_gameweek)


@cache_ignoring_session(maxsize=365)
def get_return_gameweek_by_date(
    return_date: date | datetime | str,
    team: str,
    season: str = CURRENT_SEASON,
    dbsession: Session | None = None,
) -> int:
    """
    Gameweek of a team's next match on or after `date`.

    A placeholder gameweek past the end of the season if the team has no match
    left, so that callers ordering by gameweek sort it last rather than crashing.
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
    """Gameweek of the next fixture on or after the specified date."""
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
    """The last gameweek the result table has data for."""
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
    gameweek: int | None,
    season: str,
    current_season: str = CURRENT_SEASON,
    current_gameweek: int | None = None,
) -> bool:
    """Whether this season and gameweek are at or after the current one."""
    if current_gameweek is None:
        current_gameweek = next_gameweek()
    return (
        season == current_season and (gameweek is None or gameweek >= current_gameweek)
    ) or (season != current_season and int(season) > int(current_season))


def get_gameweeks_array(
    n_gameweeks: int | None = None,
    gameweek_start: int | None = None,
    gameweek_end: int | None = None,
    season: str = CURRENT_SEASON,
    dbsession: Session | None = None,
) -> list[int]:
    """
    The given gameweeks, minus any past the end of the season.

    `gameweek_end` is inclusive, which is what `--gameweek-end` says it is
    ("Last gameweek to cover") and what `airsenal replay` has always meant by
    it. This was exclusive, so `--gameweek-start 5 --gameweek-end 10` covered
    five gameweeks under `optimize` and `predict` and six under `replay`.

    Raises:
        ValueError: None of them are still to be played.
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
        gameweek_end = gameweek_start + n_gameweeks - 1

    gameweeks = list(range(gameweek_start, gameweek_end + 1))
    max_gameweek = get_max_gameweek(season=season, dbsession=dbsession)
    gameweeks = list(filter(lambda x: x <= max_gameweek, gameweeks))

    if len(gameweeks) == 0:
        msg = "No gameweeks in specified range"
        raise ValueError(msg)
    if max(gameweeks) < gameweek_end:
        logger.warning(
            "Last gameweek set to %s (%s weeks ahead)", max(gameweeks), len(gameweeks)
        )

    return gameweeks
