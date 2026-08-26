"""Reconstructing the user's transaction history from the FPL API.

These combine the API, squad state and database writes, so they sit above all
three rather than in db/, which holds only the plain insert. state.py cannot
hold them: squad.py imports it, so anything here that builds a Squad would close
a loop.
"""

from sqlalchemy import select
from sqlalchemy.orm import Session

from airsenal.core.logging import get_logger
from airsenal.db.models import Transaction
from airsenal.db.queries.gameweeks import next_gameweek
from airsenal.db.queries.players import get_player_from_api_id
from airsenal.db.queries.transactions import (
    add_transaction,
    free_hit_used_in_gameweek,
    transaction_exists,
)
from airsenal.db.session import get_session
from airsenal.game.season import CURRENT_SEASON
from airsenal.remote.errors import RemoteError
from airsenal.remote.fpl_api import (
    FPLDataFetcher,
    get_fetcher,
    require_fpl_team_id,
)
from airsenal.squad.squad import Squad, get_current_squad_from_api
from airsenal.squad.state import get_entry_start_gameweek, get_players_for_gameweek

logger = get_logger(__name__)


def record_initial_squad_transactions(
    season: str = CURRENT_SEASON,
    tag: str = "AIrsenal" + CURRENT_SEASON,
    fpl_team_id: int | None = None,
    dbsession: Session | None = None,
) -> None:
    """
    Record an entry's opening fifteen players in the transactions table.

    The players come from the team history endpoint and their gameweek 1 prices
    from the player history endpoint. This records fifteen players that were
    already chosen; `optimization.run_squad.build_new_squad` is the one that
    runs an optimizer to choose them.
    """
    dbsession = dbsession if dbsession is not None else get_session()
    fpl_team_id = require_fpl_team_id(fpl_team_id)
    logger.info(
        "Getting initially selected players in squad %s for first gameweek...",
        fpl_team_id,
    )
    if next_gameweek() == 1:
        logger.info("Season hasn't started yet so nothing to add to the DB.")
        return

    starting_gw = get_entry_start_gameweek(fpl_team_id)
    logger.info("Got starting squad from gameweek %s.", starting_gw)
    if starting_gw == next_gameweek():
        logger.info(
            "This is team {fpl_team_id}'s first gameweek so nothing to add to the DB "
            "yet."
        )
        return

    logger.info("Adding player data...")

    init_players = get_players_for_gameweek(starting_gw, fpl_team_id)
    free_hit = free_hit_used_in_gameweek(starting_gw, fpl_team_id)
    time = get_fetcher().get_event_data()[starting_gw]["deadline"]
    for player in init_players:
        player_api_id = player.fpl_api_id
        if player_api_id is None:
            msg = f"Player {player} has no FPL API ID"
            raise ValueError(msg)
        first_gw_data = get_fetcher().get_gameweek_data_for_player(
            player_api_id, starting_gw
        )

        if len(first_gw_data) == 0:
            # Edge case where API doesn't have player data for gameweek 1, e.g. in 20/21
            # season where 4 teams didn't play gameweek 1. Calculate GW1 price from
            # API using current price and total price change.
            logger.warning(
                "Using current data to determine starting price for player %s",
                player_api_id,
            )
            pdata = get_fetcher().get_player_summary_data()[player_api_id]
            price = pdata["now_cost"] - pdata["cost_change_start"]
        else:
            price = first_gw_data[0]["value"]

        logger.info(
            "Adding player %s in GW%s for £%sm", player, starting_gw, price / 10
        )

        add_transaction(
            player.player_id,
            starting_gw,
            1,
            price,
            season,
            tag,
            free_hit,
            fpl_team_id,
            time,
            dbsession,
        )


def update_squad(
    season: str = CURRENT_SEASON,
    tag: str = "AIrsenal" + CURRENT_SEASON,
    fpl_team_id: int | None = None,
    dbsession: Session | None = None,
) -> None:
    """
    Record every transfer after gameweek 1 in the transactions table.

    From the transfers endpoint, which is the one carrying the prices actually
    paid and received.
    """
    dbsession = dbsession if dbsession is not None else get_session()
    fpl_team_id = require_fpl_team_id(fpl_team_id)
    logger.info("Updating db with squad with fpl_team_id=%s", fpl_team_id)
    # do we already have the initial squad for this fpl_team_id?
    existing_transfers = dbsession.scalars(
        select(Transaction).where(
            Transaction.fpl_team_id == fpl_team_id,
            Transaction.season == season,
        )
    ).all()
    if len(existing_transfers) == 0:
        # need to put the initial squad into the db
        record_initial_squad_transactions(
            season=season, tag=tag, fpl_team_id=fpl_team_id, dbsession=dbsession
        )
    # now update with transfers
    transfers = get_fetcher().get_fpl_transfer_data(fpl_team_id)
    for transfer in transfers:
        gameweek = transfer["event"]
        api_pid_out = transfer["element_out"]
        player_out = get_player_from_api_id(api_pid_out, dbsession=dbsession)
        if player_out is None:
            msg = f"Player with API ID {api_pid_out} not found in database."
            raise ValueError(msg)
        pid_out = player_out.player_id
        price_out = transfer["element_out_cost"]
        api_pid_in = transfer["element_in"]
        player_in = get_player_from_api_id(api_pid_in, dbsession=dbsession)
        if player_in is None:
            msg = f"Player with API ID {api_pid_in} not found in database."
            raise ValueError(msg)
        pid_in = player_in.player_id
        price_in = transfer["element_in_cost"]
        time = transfer["time"]

        if not transaction_exists(
            fpl_team_id,
            gameweek,
            season,
            time,
            pid_out,
            price_out,
            pid_in,
            price_in,
            dbsession=dbsession,
        ):
            logger.debug(
                "Adding transaction: gameweek: %s removing player %s for %s",
                gameweek,
                pid_out,
                price_out,
            )
            free_hit = free_hit_used_in_gameweek(gameweek)
            add_transaction(
                pid_out,
                gameweek,
                -1,
                price_out,
                season,
                tag,
                free_hit,
                fpl_team_id,
                time,
                dbsession,
            )

            logger.debug(
                "Adding transaction: gameweek: %s adding player %s for %s",
                gameweek,
                pid_in,
                price_in,
            )
            add_transaction(
                pid_in,
                gameweek,
                1,
                price_in,
                season,
                tag,
                free_hit,
                fpl_team_id,
                time,
                dbsession,
            )


def get_starting_squad(
    next_gw: int | None = None,
    season: str = CURRENT_SEASON,
    fpl_team_id: int | None = None,
    use_api: bool = False,
    fetcher: FPLDataFetcher | None = None,
    dbsession: Session | None = None,
) -> Squad:
    """This entry's current squad, from the transactions table or the FPL API."""
    fetcher = fetcher if fetcher is not None else get_fetcher()
    next_gw = next_gameweek() if next_gw is None else next_gw
    if use_api:
        if season != CURRENT_SEASON:
            msg = "Can only use API for current season and gameweek"
            raise RuntimeError(msg)
        if season == CURRENT_SEASON and next_gw != next_gameweek():
            msg = "Can only use API for current season and gameweek"
            raise RuntimeError(msg)
        if not fpl_team_id:
            msg = "Please specify fpl_team_id to get current squad from API"
            raise RuntimeError(msg)
        try:
            return get_current_squad_from_api(fpl_team_id, fetcher=fetcher)

        except RemoteError:
            logger.warning(
                "Failed to get current squad from API. Using DB instead, which "
                "may be out of date.",
                exc_info=True,
            )

    # otherwise, we use the Transaction table in the DB
    return get_squad_from_transactions(next_gw, season, fpl_team_id, dbsession)


def get_squad_from_transactions(
    gameweek: int | None,
    season: str = CURRENT_SEASON,
    fpl_team_id: int | None = None,
    dbsession: Session | None = None,
) -> Squad:
    """
    Rebuild the squad as it stood *before* `gameweek`, by replaying transactions.

    Only transactions strictly earlier than `gameweek` are applied, and free hit
    transfers are skipped entirely because they last a single week. Players are
    added at `gameweek` rather than at the gameweek they were bought in, so the
    squad reflects each player's current club. Budget and squad constraints are
    not checked between transfers - only the final squad has to obey them.

    With no `fpl_team_id`, the entry that made the most recent transaction is
    used.
    """
    dbsession = dbsession if dbsession is not None else get_session()
    if not fpl_team_id:
        # use the most recent transaction in the table
        most_recent = dbsession.scalars(
            select(Transaction)
            .where(Transaction.free_hit == 0, Transaction.season == season)
            .order_by(Transaction.id.desc())
            .limit(1)
        ).first()
        if most_recent is None:
            msg = "No transactions in database."
            raise ValueError(msg)
        fpl_team_id = most_recent.fpl_team_id
    logger.debug("Getting starting squad for %s", fpl_team_id)

    # Don't include free hit transfers as they only apply for the week the
    # chip is activated
    transactions = dbsession.scalars(
        select(Transaction)
        .where(
            Transaction.fpl_team_id == fpl_team_id,
            Transaction.free_hit == 0,
            Transaction.season == season,
            Transaction.gameweek < gameweek,
        )
        .order_by(Transaction.gameweek, Transaction.id)
    ).all()
    if len(transactions) == 0:
        msg = f"No transactions in database for team ID {fpl_team_id}"
        raise ValueError(msg)

    s = Squad(season=season)
    for trans in transactions:
        if trans.bought_or_sold == -1:
            s.remove_player(trans.player_id, price=trans.price)
        else:
            # within an individual transfer we can violate the budget and squad
            # constraints, as long as the final squad for that gameweek obeys them
            s.add_player(
                trans.player_id,
                price=trans.price,
                gameweek=gameweek,  # not trans.gameweek, to get player's current club
                check_budget=False,
                check_team=False,
            )
    return s
