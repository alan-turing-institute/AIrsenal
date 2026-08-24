"""Team lookups."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from airsenal.core.logging import get_logger
from airsenal.core.season import CURRENT_SEASON
from airsenal.db.models import Team
from airsenal.db.session import get_session

logger = get_logger(__name__)


def get_team_name(
    team_id: int, season: str = CURRENT_SEASON, dbsession: Session | None = None
) -> str | None:
    """
    Return 3-letter team name given a numerical id.
    These ids are based on alphabetical order of all teams in that season,
    so can vary from season to season.
    """
    dbsession = dbsession if dbsession is not None else get_session()
    team = dbsession.scalars(
        select(Team).where(Team.season == season, Team.team_id == team_id).limit(1)
    ).first()
    if team:
        return team.name
    logger.warning("Unknown team_id %s for %s season", team_id, season)
    return None


def get_teams_for_season(season: str, dbsession: Session) -> list[str]:
    """
    Query the Team table and get a list of teams for a given
    season.
    """
    teams = dbsession.scalars(select(Team).where(Team.season == season)).all()
    return [t.name for t in teams]


def database_is_empty(dbsession: Session) -> bool:
    """Whether the database has been filled yet: no teams means nothing else either."""
    return dbsession.scalars(select(Team).limit(1)).first() is None
