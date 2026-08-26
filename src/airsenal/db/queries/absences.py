"""Player absences."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from airsenal.db.models import Absence, Player
from airsenal.db.session import get_session
from airsenal.game.season import CURRENT_SEASON


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
