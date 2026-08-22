"""Player absences."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from airsenal.db.models import Absence, Player
from airsenal.db.session import get_session
from airsenal.domain.season import CURRENT_SEASON


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
