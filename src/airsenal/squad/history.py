"""Reconstructing the user's transaction history from the FPL API.

These combine the API, squad state and database writes, so they sit above all
three rather than in db/writes, which holds only the plain insert.
"""

from sqlalchemy import select
from sqlalchemy.orm import Session

from airsenal.core.logging import get_logger
from airsenal.db.models import Transaction
from airsenal.db.queries.gameweeks import next_gameweek
from airsenal.db.queries.players import get_player_from_api_id
from airsenal.db.queries.transactions import (
    free_hit_used_in_gameweek,
    transaction_exists,
)
from airsenal.db.session import get_session
from airsenal.db.writes.transactions import add_transaction
from airsenal.domain.season import CURRENT_SEASON
from airsenal.fetch.fpl_api import get_fetcher
from airsenal.squad.state import get_entry_start_gameweek, get_players_for_gameweek

logger = get_logger(__name__)


def fill_initial_squad(
    season=CURRENT_SEASON,
    tag="AIrsenal" + CURRENT_SEASON,
    fpl_team_id=None,
    dbsession: Session | None = None,
):
    """
    Fill the Transactions table in the database with the initial 15 players, and their
    costs, getting the information from the team history API endpoint (for the list of
    players in our team) and the player history API endpoint (for their price in gw1).
    """

    dbsession = dbsession if dbsession is not None else get_session()
    if not fpl_team_id:
        fpl_team_id = get_fetcher().FPL_TEAM_ID
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
    season=CURRENT_SEASON,
    tag="AIrsenal" + CURRENT_SEASON,
    fpl_team_id=None,
    dbsession: Session | None = None,
):
    """
    Fill the Transactions table in the DB with all the transfers in gameweeks after 1,
    using the transfers API endpoint which has the correct buy and sell prices.
    """
    dbsession = dbsession if dbsession is not None else get_session()
    if not fpl_team_id:
        fpl_team_id = get_fetcher().FPL_TEAM_ID
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
        fill_initial_squad(
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
