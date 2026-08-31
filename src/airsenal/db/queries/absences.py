"""Player absences."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from airsenal.core.caching import cache_ignoring_session
from airsenal.db.models import Absence, Player
from airsenal.db.session import get_session
from airsenal.game.season import CURRENT_SEASON


@cache_ignoring_session(maxsize=2048)
def absence_gameweeks(
    player_id: int, season: str, dbsession: Session | None = None
) -> tuple[tuple[int, int], ...]:
    """
    The (from, until) gameweek ranges a player was absent for in a past season.

    Half-open: `gw_from` is the first gameweek missed and `gw_until` the gameweek
    they returned in, so the two are equal when nothing was missed at all.

    One query per player per season rather than one per player per fixture: this
    is read from the innermost loop of the points prediction, and over a whole
    replay that was tens of thousands of queries for an answer that does not
    change within a season. Ranges with no end gameweek are left out, matching
    the SQL this replaced, where a NULL `gw_until` never satisfied the comparison.
    """
    dbsession = dbsession if dbsession is not None else get_session()
    absences = dbsession.scalars(
        select(Absence).where(
            Absence.season == season,
            Absence.player_id == player_id,
        )
    ).all()
    return tuple((a.gw_from, a.gw_until) for a in absences if a.gw_until is not None)


def was_historic_absence(
    player: Player, gameweek: int, season: str, dbsession: Session | None = None
) -> bool:
    """
    Whether a player was injured or suspended in a past gameweek.

    Always False for the current season - the Absence table only covers seasons
    that have finished, and the FPL API is what says who is out now.
    """
    if season == CURRENT_SEASON:
        # we only consider past seasons here
        return False
    # `gw_from <=`, not `<`: ingest resolves it to the team's next match on or
    # after the day the absence began, so it is the first gameweek missed rather
    # than the last one played. Excluding it called the opening week of every
    # absence available, and that is the one week the recent-minutes guard in
    # `prediction/points.py` cannot catch either - the minutes it reads all
    # predate the absence.
    return any(
        gw_from <= gameweek < gw_until
        for gw_from, gw_until in absence_gameweeks(
            player.player_id, season, dbsession=dbsession
        )
    )
