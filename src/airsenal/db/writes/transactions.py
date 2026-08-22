"""Recording squad transactions."""

from sqlalchemy.orm import Session

from airsenal.core.logging import get_logger
from airsenal.db.models import Transaction
from airsenal.db.session import get_session

logger = get_logger(__name__)


def add_transaction(
    player_id,
    gameweek,
    in_or_out,
    price,
    season,
    tag,
    free_hit,
    fpl_team_id,
    time,
    dbsession: Session | None = None,
):
    """
    add buy (in_or_out=1) or sell (in_or_out=-1) transactions to the db table.
    """
    dbsession = dbsession if dbsession is not None else get_session()
    t = Transaction(
        player_id=player_id,
        gameweek=gameweek,
        bought_or_sold=in_or_out,
        price=price,
        season=season,
        tag=tag,
        free_hit=free_hit,
        fpl_team_id=fpl_team_id,
        time=time,
    )
    dbsession.add(t)
    dbsession.commit()
