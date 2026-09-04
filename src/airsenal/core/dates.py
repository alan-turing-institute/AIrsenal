"""Date and datetime parsing helpers."""

from datetime import date, datetime
from functools import lru_cache

import dateparser
from dateutil.parser import isoparse


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
