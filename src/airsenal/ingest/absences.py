"""
Loading player absences (injuries, suspensions) from the packaged CSV.

The counterpart to `export/absences.py`. Each row gives a date range, which is
resolved to a half-open range of gameweeks: `gw_from` is the first one the
absence could have kept the player out of, and `gw_until` the one they were back
for. The two are equal when the absence cost them no match at all.

This is primarily for the scraped Transfermarkt data. The FPL API statuses are saved in
the player attributes history files.
"""

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import pandas as pd
from sqlalchemy.orm.session import Session

from airsenal.core.console import track
from airsenal.core.data_files import FilePath, absences_file
from airsenal.core.logging import get_logger
from airsenal.db.models import Absence
from airsenal.db.queries.gameweeks import (
    get_gameweek_by_date,
    get_max_gameweek,
    get_return_gameweek_by_date,
)
from airsenal.db.queries.players import get_player, get_player_by_similar_name
from airsenal.db.session import get_session
from airsenal.game.season import CURRENT_SEASON, get_past_seasons, sort_seasons

if TYPE_CHECKING:
    from datetime import date

    from airsenal.db.models import Player

logger = get_logger(__name__)


def gameweek_returned(
    date_until: "date", player: "Player", season: str, dbsession: Session
) -> int:
    """
    The gameweek a player was available again, from the day their absence ended.

    One past the season's last gameweek when the end date is past its last
    fixture, so that the half-open range covers the rest of the season. Readers skip an
    absence with no end gameweek, so leaving it unresolved would ignore the absence
    entirely.
    """
    gameweek = get_gameweek_by_date(
        check_date=date_until, season=season, dbsession=dbsession
    )
    if gameweek is None:
        return get_max_gameweek(season, dbsession=dbsession) + 1
    return get_return_gameweek_by_date(
        date_until, player.team(gameweek, season), season, dbsession=dbsession
    )


def load_absences(
    season: str, dbsession: Session, path: FilePath | None = None
) -> None:
    logger.info("ABSENCES %s", season)
    if path is None:
        path = absences_file(season)
    absences = pd.read_csv(path, parse_dates=["from", "until"])

    for _, row in track(
        absences.iterrows(), total=absences.shape[0], description=f"ABSENCES {season}"
    ):
        # Two thirds of the names the exact lookup misses are academy players
        # who never reach the FPL game at all; the rest are spelled differently
        # by whoever the row came from.
        p = get_player(row["player"], dbsession=dbsession)
        if p is None:
            p = get_player_by_similar_name(row["player"], dbsession=dbsession)
        if not p:
            logger.warning("Couldn't find player %s", row["player"])
            continue

        date_from = row["from"].date()
        if date_from is pd.NaT:
            logger.warning("%s %s has no from date", row["player"], row["details"])
            continue

        # first check approx gameweek to determine player's team at that time
        gw_date = get_gameweek_by_date(
            check_date=date_from, season=season, dbsession=dbsession
        )
        if gw_date is None:
            logger.warning(
                "Couldn't find gameweek for %s from date %s", row["player"], date_from
            )
            continue
        team_from = p.team(gw_date, season)
        # The first gameweek the absence could have stopped them playing, being
        # the first of their team's matches to kick off *after* it began - hence
        # the day after, rather than `date_from` itself.
        gw_from = get_return_gameweek_by_date(
            date_from + timedelta(days=1), team_from, season, dbsession=dbsession
        )

        date_until = None if row["until"] is pd.NaT else row["until"].date()
        gw_until = (
            None
            if date_until is None
            else gameweek_returned(date_until, p, season, dbsession=dbsession)
        )

        url = row["url"]
        timestamp = datetime.now().isoformat()
        absence = Absence(
            player=p,
            player_id=p.player_id,
            season=season,
            reason=row["reason"],
            details=row["details"],
            # These columns are VARCHAR, so write ISO-8601 text rather than date objects
            date_from=date_from.isoformat(),
            date_until=date_until.isoformat() if date_until is not None else None,
            gw_from=gw_from,
            gw_until=gw_until,
            url=url,
            timestamp=timestamp,
        )
        dbsession.add(absence)
    dbsession.commit()


def make_absence_table(
    seasons: list[str] | None = None, dbsession: Session | None = None
) -> None:
    dbsession = dbsession if dbsession is not None else get_session()
    if seasons is None:
        seasons = []
    if not seasons:
        seasons = [CURRENT_SEASON]
        seasons += get_past_seasons(3)
    for season in sort_seasons(seasons):
        if season == CURRENT_SEASON:
            continue
        load_absences(season, dbsession)
