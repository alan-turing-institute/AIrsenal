"""Reading and recording the transaction history."""

from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session

from airsenal.core.logging import get_logger
from airsenal.db.models import Transaction
from airsenal.db.session import get_session
from airsenal.fetch.fpl_api import get_fetcher

logger = get_logger(__name__)


def free_hit_used_in_gameweek(gameweek, fpl_team_id=None):
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


def count_transactions(season, fpl_team_id, dbsession: Session | None = None):
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
    fpl_team_id,
    gameweek,
    season,
    time,
    pid_out,
    price_out,
    pid_in,
    price_in,
    dbsession: Session | None = None,
):
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
