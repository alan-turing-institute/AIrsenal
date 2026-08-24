"""Reading and recording the transaction history."""

from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session

from airsenal.core.logging import get_logger
from airsenal.db.models import Transaction
from airsenal.db.session import get_session
from airsenal.remote.fpl_api import get_fetcher

logger = get_logger(__name__)


def free_hit_used_in_gameweek(gameweek: int, fpl_team_id: int | None = None) -> int:
    """Use FPL API to determine whether a chip was played in the given gameweek"""
    if not fpl_team_id:
        fpl_team_id = get_fetcher().FPL_TEAM_ID
    fpl_team_data = get_fetcher().get_fpl_team_data(gameweek, fpl_team_id)
    if (
        fpl_team_data
        and "active_chip" in fpl_team_data
        and fpl_team_data["active_chip"] == "freehit"
    ):
        return 1
    return 0


def count_transactions(
    season: str, fpl_team_id: int | None, dbsession: Session | None = None
) -> int:
    """Count the number of transactions we have in the database for a given team ID
    and season.
    """
    dbsession = dbsession if dbsession is not None else get_session()
    if fpl_team_id is None:
        fpl_team_id = get_fetcher().FPL_TEAM_ID

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
    fpl_team_id: int,
    gameweek: int,
    season: str,
    time: str,
    pid_out: int,
    price_out: int,
    pid_in: int,
    price_in: int,
    dbsession: Session | None = None,
) -> bool:
    """Check whether the transactions related to transferring a player in and out
    in a gameweek at a specific time already exist in the database.
    """
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
    gameweek: int,
    in_or_out: int,
    price: int,
    season: str,
    tag: str,
    free_hit: int,
    fpl_team_id: int,
    time: str,
    dbsession: Session | None = None,
) -> None:
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
