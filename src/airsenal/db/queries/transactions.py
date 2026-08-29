"""Reading and recording the transaction history."""

from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session

from airsenal.core.logging import get_logger
from airsenal.db.models import Transaction
from airsenal.db.session import get_session

logger = get_logger(__name__)


def count_transactions(
    season: str, fpl_team_id: int, dbsession: Session | None = None
) -> int:
    """How many transactions the database holds for a team in a season."""
    dbsession = dbsession if dbsession is not None else get_session()
    return (
        dbsession.scalar(
            select(func.count(Transaction.id)).where(
                Transaction.fpl_team_id == fpl_team_id,
                Transaction.season == season,
            )
        )
        or 0
    )


def transaction_exists(
    gameweek: int,
    season: str,
    fpl_team_id: int,
    time: str,
    pid_out: int,
    price_out: int,
    pid_in: int,
    price_in: int,
    dbsession: Session | None = None,
) -> bool:
    """Whether both halves of this transfer are already recorded."""
    dbsession = dbsession if dbsession is not None else get_session()
    transaction_count = (
        dbsession.scalar(
            select(func.count(Transaction.id)).where(
                Transaction.fpl_team_id == fpl_team_id,
                Transaction.gameweek == gameweek,
                Transaction.season == season,
                Transaction.time == time,
                or_(
                    and_(
                        Transaction.player_id == pid_in,
                        Transaction.price == price_in,
                        Transaction.bought_or_sold == 1,
                    ),
                    and_(
                        Transaction.player_id == pid_out,
                        Transaction.price == price_out,
                        Transaction.bought_or_sold == -1,
                    ),
                ),
            )
        )
        or 0
    )
    if transaction_count == 2:  # row for player bought and player sold
        return True
    if transaction_count == 0:
        return False
    msg = (
        f"Database error: {transaction_count} transactions in the database with "
        f"parameters:  fpl_team_id={fpl_team_id}, gameweek={gameweek}, "
        f"time={time}, pid_in={pid_in}, pid_out={pid_out}. Should be 2."
    )
    raise ValueError(msg)


def add_transaction(
    player_id: int,
    tag: str,
    gameweek: int,
    in_or_out: int,
    price: int,
    season: str,
    free_hit: int,
    fpl_team_id: int,
    time: str,
    dbsession: Session | None = None,
) -> None:
    """Record a buy (in_or_out=1) or a sell (in_or_out=-1)."""
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
