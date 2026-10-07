"""Fill the "player_risk" table from the FPL API's `scout_risks`."""

from sqlalchemy import delete, or_, select
from sqlalchemy.orm.session import Session

from airsenal.core.logging import get_logger
from airsenal.db.models import PlayerRisk
from airsenal.db.queries.gameweeks import next_gameweek
from airsenal.db.queries.players import get_player_from_api_id
from airsenal.db.session import get_session
from airsenal.remote.fpl_api import get_fetcher

logger = get_logger(__name__)


def fill_player_risks_from_api(season: str, dbsession: Session | None = None) -> None:
    """
    Record the gameweeks the FPL API says players will miss, for the current season.

    The API drops a risk once its gameweek has passed, and can withdraw one that has
    not, so the rows from the next gameweek on are replaced with what it says now
    and the earlier ones are kept.
    """
    dbsession = get_session(dbsession)
    gameweek = next_gameweek()
    dbsession.execute(
        delete(PlayerRisk).where(
            PlayerRisk.season == season,
            or_(PlayerRisk.gameweek.is_(None), PlayerRisk.gameweek >= gameweek),
        )
    )
    kept = {
        (row.player_id, row.gameweek, row.property)
        for row in dbsession.scalars(
            select(PlayerRisk).where(PlayerRisk.season == season)
        )
    }

    n_added = 0
    for player_api_id, p_summary in get_fetcher().get_player_summary_data().items():
        risks = p_summary.get("scout_risks") or []
        if not risks:
            continue
        player = get_player_from_api_id(player_api_id, dbsession=dbsession)
        if not player:
            logger.warning("RISKS %s No player found with id %s", season, player_api_id)
            continue
        for risk in risks:
            key = (player.player_id, risk.get("gameweek"), risk["property"])
            if key in kept:
                continue
            kept.add(key)
            dbsession.add(
                PlayerRisk(
                    player_id=player.player_id,
                    season=season,
                    gameweek=risk.get("gameweek"),
                    property=risk["property"],
                    notes=risk.get("notes"),
                )
            )
            n_added += 1
    dbsession.commit()
    logger.info("RISKS %s: %s recorded", season, n_added)
