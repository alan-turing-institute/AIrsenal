"""
Loading player absences (injuries, suspensions) from the packaged CSV.

Each row gives a date range, which is resolved to a half-open range of gameweeks:
the first one the absence could have kept the player out of, and the one they were
back for. The two are equal when the absence cost them no match at all.

This is the fallback source of availability, for the seasons and gameweeks the
per-day attributes history does not reach. Where it does reach,
`ingest.attributes_history` is the better answer, because it records what the FPL
API said at the time rather than what Transfermarkt says with hindsight.
"""

from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from sqlalchemy.orm.session import Session

from airsenal.core.console import track
from airsenal.core.data_files import FilePath, absences_file
from airsenal.core.logging import get_logger
from airsenal.db.queries.gameweeks import (
    get_gameweek_by_date,
    get_max_gameweek,
    get_return_gameweek_by_date,
)
from airsenal.db.queries.players import get_player, get_player_by_similar_name
from airsenal.ingest.attributes_history import Availability

if TYPE_CHECKING:
    from datetime import date

    from airsenal.db.models import Player

logger = get_logger(__name__)


def gameweek_returned(
    date_until: "date | None", player: "Player", season: str, dbsession: Session
) -> int:
    """
    The gameweek a player was available again, from the day their absence ended.

    One past the season's last gameweek when there is no end date, or when it is
    past the last fixture, so that the half-open range covers the rest of the
    season. Every packaged file is a scrape of a finished season, where an absence
    with no recorded end is one the player did not come back from.
    """
    end_of_season = get_max_gameweek(season, dbsession=dbsession) + 1
    if date_until is None:
        return end_of_season
    gameweek = get_gameweek_by_date(
        check_date=date_until, season=season, dbsession=dbsession
    )
    if gameweek is None:
        return end_of_season
    return get_return_gameweek_by_date(
        date_until, player.team(gameweek, season), season, dbsession=dbsession
    )


def resolve_absence_gameweeks(
    row: "pd.Series[Any]", season: str, dbsession: Session
) -> "tuple[Player, int, int] | None":
    """
    The player and half-open gameweek range one row of the absences csv covers.

    Returns:
        None if the row names nobody we know, or has no usable start date.
    """
    # Most names the exact lookup misses are academy players who never reach
    # the FPL game; the rest are spelled differently.
    player = get_player(row["player"], dbsession=dbsession)
    if player is None:
        player = get_player_by_similar_name(row["player"], dbsession=dbsession)
    if not player:
        logger.warning("Couldn't find player %s", row["player"])
        return None

    if row["from"] is pd.NaT:
        logger.warning("%s %s has no from date", row["player"], row["details"])
        return None
    date_from = row["from"].date()

    # first check approx gameweek to determine player's team at that time
    gameweek_date = get_gameweek_by_date(
        check_date=date_from, season=season, dbsession=dbsession
    )
    if gameweek_date is None:
        logger.warning(
            "Couldn't find gameweek for %s from date %s", row["player"], date_from
        )
        return None
    team_from = player.team(gameweek_date, season)
    # The first gameweek the absence could have stopped them playing, being
    # the first of their team's matches to kick off *after* it began - hence
    # the day after, rather than `date_from` itself.
    gameweek_from = get_return_gameweek_by_date(
        date_from + timedelta(days=1), team_from, season, dbsession=dbsession
    )

    date_until = None if row["until"] is pd.NaT else row["until"].date()
    gameweek_until = gameweek_returned(date_until, player, season, dbsession=dbsession)
    return player, gameweek_from, gameweek_until


def absence_news(row: "pd.Series[Any]") -> str:
    """What to record as the news for an absence, from its csv row."""
    details = row["details"]
    news = row["reason"] if pd.isna(details) else details
    # `news` is a str100 column
    return str(news)[:100]


def get_availability_from_absences(
    season: str, dbsession: Session, path: FilePath | None = None
) -> dict[tuple[int, int], Availability]:
    """
    Availability per (player id, gameweek), inferred from the packaged absences csv.

    A gameweek inside an absence reads as a 0% chance of playing, due back in the
    gameweek the player returned - the shape `Player.is_injured_or_suspended` asks of
    the FPL API's own flags.
    """
    logger.info("ABSENCES %s", season)
    if path is None:
        path = absences_file(season)
    if not Path(path).is_file():
        # A season only gets a file once it has been scraped, which for the season
        # in progress is usually not yet.
        logger.info("No absences file at %s", path)
        return {}
    absences = pd.read_csv(path, parse_dates=["from", "until"])

    availability: dict[tuple[int, int], Availability] = {}
    for _, row in track(
        absences.iterrows(), total=absences.shape[0], description=f"ABSENCES {season}"
    ):
        resolved = resolve_absence_gameweeks(row, season, dbsession)
        if resolved is None:
            continue
        player, gameweek_from, gameweek_until = resolved
        news = absence_news(row)
        for gameweek in range(gameweek_from, gameweek_until):
            key = (player.player_id, gameweek)
            existing = availability.get(key)
            existing_until = existing.return_gameweek if existing is not None else None
            # Overlapping absences: the one that keeps them out longest wins, news
            # and all, so that the two describe the same absence.
            if existing_until is not None and existing_until >= gameweek_until:
                continue
            availability[key] = Availability(
                news=news,
                chance_of_playing_next_round=0,
                return_gameweek=gameweek_until,
            )
    return availability
