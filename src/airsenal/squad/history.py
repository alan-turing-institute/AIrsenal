"""Reconstructing the user's transaction history from the FPL API."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from airsenal.core.logging import get_logger
from airsenal.db.models import Transaction
from airsenal.db.queries.gameweeks import next_gameweek
from airsenal.db.queries.players import require_api_id, require_player_from_api_id
from airsenal.db.queries.transactions import (
    add_transaction,
    transaction_exists,
)
from airsenal.db.session import get_session
from airsenal.game.enums import Chip
from airsenal.game.season import CURRENT_SEASON
from airsenal.remote.errors import RemoteError
from airsenal.remote.fpl_api import (
    FPLDataFetcher,
    get_fetcher,
    require_fpl_team_id,
)
from airsenal.squad.squad import Squad, get_current_squad_from_api
from airsenal.squad.state import (
    chip_used_in_gameweek,
    get_entry_start_gameweek,
    get_players_for_gameweek,
)

logger = get_logger(__name__)


def record_initial_squad_transactions(
    tag: str = "AIrsenal" + CURRENT_SEASON,
    season: str = CURRENT_SEASON,
    fpl_team_id: int | None = None,
    dbsession: Session | None = None,
) -> None:
    """
    Record an entry's opening fifteen players in the transactions table.

    The players come from the team history endpoint and their gameweek 1 prices
    from the player history endpoint.
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

    starting_gameweek = get_entry_start_gameweek(fpl_team_id)
    logger.info("Got starting squad from gameweek %s.", starting_gameweek)
    if starting_gameweek == next_gameweek():
        logger.info(
            "This is team %s's first gameweek so nothing to add to the DB yet.",
            fpl_team_id,
        )
        return

    logger.info("Adding player data...")

    init_players = get_players_for_gameweek(starting_gameweek, fpl_team_id)
    free_hit = int(
        chip_used_in_gameweek(starting_gameweek, fpl_team_id) is Chip.FREE_HIT
    )
    time = get_fetcher().get_event_data()[starting_gameweek]["deadline"]
    for player in init_players:
        player_api_id = require_api_id(player)
        first_gameweek_data = get_fetcher().get_gameweek_data_for_player(
            player_api_id, starting_gameweek
        )

        if len(first_gameweek_data) == 0:
            # The API has no data for a player whose team did not play the
            # starting gameweek (four teams missed gameweek 1 in 20/21), so the
            # price is worked back from the current price and the change since.
            logger.warning(
                "Using current data to determine starting price for player %s",
                player_api_id,
            )
            pdata = get_fetcher().get_player_summary_data()[player_api_id]
            price = pdata["now_cost"] - pdata["cost_change_start"]
        else:
            price = first_gameweek_data[0]["value"]

        logger.info(
            "Adding player %s in GW%s for £%sm", player, starting_gameweek, price / 10
        )

        add_transaction(
            player.player_id,
            tag,
            starting_gameweek,
            in_or_out=1,
            price=price,
            season=season,
            free_hit=free_hit,
            # The squad an entry is given to start with cost it no transfers.
            counts_as_transfer=0,
            fpl_team_id=fpl_team_id,
            time=time,
            dbsession=dbsession,
        )


def update_squad(
    tag: str = "AIrsenal" + CURRENT_SEASON,
    season: str = CURRENT_SEASON,
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
        record_initial_squad_transactions(
            season=season, tag=tag, fpl_team_id=fpl_team_id, dbsession=dbsession
        )
    transfers = get_fetcher().get_fpl_transfer_data(fpl_team_id)
    for transfer in transfers:
        gameweek = transfer["event"]
        api_pid_out = transfer["element_out"]
        pid_out = require_player_from_api_id(api_pid_out, dbsession=dbsession).player_id
        price_out = transfer["element_out_cost"]
        api_pid_in = transfer["element_in"]
        pid_in = require_player_from_api_id(api_pid_in, dbsession=dbsession).player_id
        price_in = transfer["element_in_cost"]
        time = transfer["time"]

        if not transaction_exists(
            gameweek,
            season,
            fpl_team_id,
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
            chip = chip_used_in_gameweek(gameweek, fpl_team_id)
            free_hit = int(chip is Chip.FREE_HIT)
            # A wildcard or free hit makes the gameweek's transfers unlimited.
            counts_as_transfer = int(chip is None or not chip.rebuilds_squad)
            add_transaction(
                pid_out,
                tag,
                gameweek,
                in_or_out=-1,
                price=price_out,
                season=season,
                free_hit=free_hit,
                counts_as_transfer=counts_as_transfer,
                fpl_team_id=fpl_team_id,
                time=time,
                dbsession=dbsession,
            )

            logger.debug(
                "Adding transaction: gameweek: %s adding player %s for %s",
                gameweek,
                pid_in,
                price_in,
            )
            add_transaction(
                pid_in,
                tag,
                gameweek,
                in_or_out=1,
                price=price_in,
                season=season,
                free_hit=free_hit,
                counts_as_transfer=counts_as_transfer,
                fpl_team_id=fpl_team_id,
                time=time,
                dbsession=dbsession,
            )


def get_starting_squad(
    gameweek: int | None = None,
    season: str = CURRENT_SEASON,
    fpl_team_id: int | None = None,
    use_api: bool = False,
    fetcher: FPLDataFetcher | None = None,
    dbsession: Session | None = None,
) -> Squad:
    """This entry's current squad, from the transactions table or the FPL API."""
    fetcher = fetcher if fetcher is not None else get_fetcher()
    gameweek = next_gameweek() if gameweek is None else gameweek
    if use_api:
        if season != CURRENT_SEASON or gameweek != next_gameweek():
            msg = "Can only use API for current season and gameweek"
            raise RuntimeError(msg)
        if not fpl_team_id:
            msg = "Please specify fpl_team_id to get current squad from API"
            raise RuntimeError(msg)
        try:
            return get_current_squad_from_api(fpl_team_id=fpl_team_id, fetcher=fetcher)

        except RemoteError:
            logger.warning(
                "Failed to get current squad from API. Using DB instead, which "
                "may be out of date.",
                exc_info=True,
            )

    return get_squad_from_transactions(gameweek, season, fpl_team_id, dbsession)


def get_squad_from_transactions(
    gameweek: int | None,
    season: str = CURRENT_SEASON,
    fpl_team_id: int | None = None,
    dbsession: Session | None = None,
) -> Squad:
    """
    Rebuild the squad as it stood *before* `gameweek`, by replaying transactions.

    Only transactions strictly earlier than `gameweek` are applied, and free hit
    transfers are skipped entirely because they last a single gameweek. Players are
    added at `gameweek` rather than at the gameweek they were bought in, so the
    squad reflects each player's current club. Budget and squad constraints are
    not checked between transfers - only the final squad has to obey them.
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
            s.add_player(
                trans.player_id,
                price=trans.price,
                gameweek=gameweek,  # not trans.gameweek, to get player's current club
                check_budget=False,
                check_team=False,
            )
    return s
