import os
from datetime import datetime

import pandas as pd
from sqlalchemy.orm.session import Session

from airsenal.framework.output import get_logger, track
from airsenal.framework.schema import Absence, get_session
from airsenal.framework.season import CURRENT_SEASON, sort_seasons
from airsenal.framework.utils import (
    get_gameweek_by_date,
    get_past_seasons,
    get_player,
    get_return_gameweek_by_date,
)

logger = get_logger(__name__)


def get_absences_path(season: str) -> str:
    """Path of the absences csv file for a season."""
    return os.path.join(
        os.path.dirname(__file__), "..", "data", f"absences_{season}.csv"
    )


def load_absences(season: str, dbsession: Session, path: str | None = None) -> None:
    logger.info("ABSENCES %s", season)
    if path is None:
        path = get_absences_path(season)
    absences = pd.read_csv(path, parse_dates=["from", "until"])

    for _, row in track(
        absences.iterrows(), total=absences.shape[0], description=f"ABSENCES {season}"
    ):
        p = get_player(row["player"], dbsession=dbsession)
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
        team_from = p.team(season, gw_date)
        # then get actual return gameweek using the player's team
        gw_from = get_return_gameweek_by_date(date_from, team_from, season, dbsession)

        date_until = None if row["until"] is pd.NaT else row["until"].date()
        if date_until is not None and (
            gw_date := get_gameweek_by_date(
                check_date=date_until, season=season, dbsession=dbsession
            )
        ):
            team_until = p.team(season, gw_date)
            gw_until = get_return_gameweek_by_date(
                date_until, team_until, season, dbsession
            )
        else:
            gw_until = None

        url = row["url"]
        timestamp = datetime.now().isoformat()
        absence = Absence(
            player=p,
            player_id=p.player_id,
            season=season,
            reason=row["reason"],
            details=row["details"],
            # These columns are VARCHAR, so write ISO-8601 text rather than date
            # objects. Passing a date relied on sqlite3's default date adapter, which
            # is deprecated in Python 3.12 and produces exactly this string anyway.
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
