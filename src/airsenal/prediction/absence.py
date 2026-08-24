"""Reading return dates out of FPL news text."""

import dateparser
import regex as re
from sqlalchemy.orm import Session

from airsenal.core.season import CURRENT_SEASON
from airsenal.db.queries.gameweeks import get_return_gameweek_by_date
from airsenal.db.session import get_session


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
