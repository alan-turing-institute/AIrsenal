"""Historical player absences."""

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

    Half-open: `gameweek_from` is the first gameweek missed and `gameweek_until`
    the gameweek they returned in, so the two are equal when nothing was missed
    at all.

    One query per player per season rather than one per player per fixture: this
    is read from the innermost loop of the points prediction, and over a whole
    replay that was tens of thousands of queries for an answer that does not
    change within a season.
    """
    dbsession = dbsession if dbsession is not None else get_session()
    absences = dbsession.scalars(
        select(Absence).where(
            Absence.season == season,
            Absence.player_id == player_id,
        )
    ).all()
    return tuple(
        (a.gameweek_from, a.gameweek_until)
        for a in absences
        if a.gameweek_until is not None
    )


def was_historic_absence(
    player: Player,
    current_gameweek: int,
    fixture_gameweek: int,
    season: str,
    dbsession: Session | None = None,
) -> bool:
    """
    Whether a player was already absent, and still absent for a later fixture.

    The two gameweeks are different points in time, as in
    `Player.is_injured_or_suspended`, which this is the past-season counterpart of:
    `current_gameweek` is when we are asking, `fixture_gameweek` the fixture we
    are asking about. An absence beginning after `current_gameweek` has not
    happened yet, so it does not count - a replay must not rule a player out of a
    gameweek on the strength of an injury they had not picked up at the time.

    Always False for the current season - the Absence table only covers seasons
    that have finished, and the FPL API is what says who is out now.
    """
    if season == CURRENT_SEASON:
        return False
    # `gameweek_from <=`, not `<`: ingest resolves it to the team's next match on or
    # after the day the absence began, so it is the first gameweek missed rather
    # than the last one played.
    return any(
        gameweek_from <= current_gameweek and fixture_gameweek < gameweek_until
        for gameweek_from, gameweek_until in absence_gameweeks(
            player.player_id, season, dbsession=dbsession
        )
    )
