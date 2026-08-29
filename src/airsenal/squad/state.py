"""The state of the user's own squad, combining the database and the FPL API."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from airsenal.core.logging import get_logger
from airsenal.db.models import Player, Transaction
from airsenal.db.queries.gameweeks import next_gameweek
from airsenal.db.queries.players import get_player_from_api_id
from airsenal.db.session import get_session
from airsenal.game.season import CURRENT_SEASON
from airsenal.remote.errors import (
    RemoteConnectionError,
    RemoteError,
    RemoteHTTPError,
)
from airsenal.remote.fpl_api import FPLDataFetcher, get_fetcher

logger = get_logger(__name__)


def get_bank(
    gameweek: int | None = None,
    season: str = CURRENT_SEASON,
    fpl_team_id: int | None = None,
    fetcher: FPLDataFetcher | None = None,
) -> int:
    """
    How much this entry had in the bank before a gameweek.

    `gameweek` defaults to the most recent, and `fpl_team_id` to `$FPL_TEAM_ID`.
    """
    fetcher = fetcher if fetcher is not None else get_fetcher()
    if season != CURRENT_SEASON:
        msg = "Calculating the bank for past seasons not yet implemented"
        raise RuntimeError(msg)

    if not fpl_team_id:
        fpl_team_id = get_fetcher().FPL_TEAM_ID
    # check if we're logged in, which will let us get the most up-to-date info
    try:
        return fetcher.get_current_bank(fpl_team_id)
    except RemoteError:
        logger.warning(
            "Failed to get actual bank from a logged in API. "
            "Will try to estimate it from the API without logging in, which will "
            "not include any transfers made in the current gameweek.",
            exc_info=True,
        )
        data = fetcher.get_fpl_team_history_data(fpl_team_id)
        if "current" not in data or len(data["current"]) <= 0:
            return 0

        if gameweek and isinstance(gameweek, int):
            for gw in data["current"]:
                if gw["event"] == gameweek - 1:  # value after previous gameweek
                    return int(gw["bank"])
        # otherwise, return the most recent value
        return int(data["current"][-1]["bank"])


def get_entry_start_gameweek(
    fpl_team_id: int, fetcher: FPLDataFetcher | None = None
) -> int:
    """The gameweek an entry joined, being the first the API has picks for."""
    fetcher = fetcher if fetcher is not None else get_fetcher()
    starting_gw = 1
    while starting_gw < next_gameweek():
        try:
            if get_players_for_gameweek(starting_gw, fpl_team_id, fetcher=fetcher):
                return starting_gw
            starting_gw += 1
        except RemoteHTTPError:
            starting_gw += 1
        except RemoteConnectionError:
            logger.warning(
                "Failed to connect to the API. Assuming team %s"
                " was entered in GW1 which may be incorrect.",
                fpl_team_id,
                exc_info=True,
            )
            return 1

    # if we failed to find picks in any gameweek, or we're before the start of the
    # season, assume this team ID was entered in NEXT_GAMEWEEK
    return next_gameweek()


def get_free_transfers(
    gameweek: int | None = None,
    season: str = CURRENT_SEASON,
    fpl_team_id: int | None = None,
    fetcher: FPLDataFetcher | None = None,
    dbsession: Session | None = None,
    is_replay: bool = False,
) -> int:
    """
    How many free transfers this entry had before a gameweek.

    `gameweek` defaults to the most recent, and `fpl_team_id` to `$FPL_TEAM_ID`.
    """
    fetcher = fetcher if fetcher is not None else get_fetcher()
    dbsession = dbsession if dbsession is not None else get_session()
    if season == CURRENT_SEASON and not is_replay:
        # we will use the API to estimate num transfers
        resolved_fpl_team_id = (
            fpl_team_id if fpl_team_id is not None else fetcher.FPL_TEAM_ID
        )
        if resolved_fpl_team_id is None:
            msg = "FPL team ID is required to estimate free transfers from the API"
            raise RuntimeError(msg)

        # try to get the most up-to-date info from logged in api
        try:
            return fetcher.get_num_free_transfers(resolved_fpl_team_id)
        except RemoteError:
            logger.warning(
                "Failed to get actual free transfers from a logged in API. "
                "Will try to estimate it from the API without logging in, which will "
                "not include any transfers used in the current gameweek.",
                exc_info=True,
            )
        # try to calculate free transfers based on previous transfer history in API
        try:
            data = fetcher.get_fpl_team_history_data(resolved_fpl_team_id)
            num_free_transfers = 1
            if "current" in data and len(data["current"]) > 0:
                starting_gw = get_entry_start_gameweek(
                    resolved_fpl_team_id, fetcher=fetcher
                )
                for gw in data["current"]:
                    if gw["event"] <= starting_gw:
                        continue
                    if gw["event_transfers"] == 0 and num_free_transfers < 2:
                        num_free_transfers += 1
                    elif gw["event_transfers"] >= 2:
                        num_free_transfers = 1
                    # if gameweek was specified, and we reached the previous one,
                    # break out of loop.
                    if gameweek and gw["event"] == gameweek - 1:
                        break
            return num_free_transfers
        except RemoteError:
            logger.warning(
                "Failed to estimate free transfers from the API. "
                "Will estimate from the DB instead, which may be out of date.",
                exc_info=True,
            )

    # historical/simulated data or API failed - fetch from database
    transactions = dbsession.scalars(
        select(Transaction)
        .where(Transaction.fpl_team_id == fpl_team_id, Transaction.bought_or_sold == 1)
        .order_by(Transaction.gameweek, Transaction.id)
    ).all()
    if len(transactions) == 0:
        return 1
    starting_gw = transactions[0].gameweek
    gw_transactions = {}
    for t in transactions:
        if t.gameweek not in gw_transactions:
            gw_transactions[t.gameweek] = 0
        gw_transactions[t.gameweek] += 1
    num_free_transfers = 1
    if gameweek is None and (season != CURRENT_SEASON or is_replay):
        msg = "Gameweek must be specified for historical data"
        raise ValueError(msg)
    gameweek = gameweek or next_gameweek()
    for prev_gw in range(starting_gw + 1, gameweek):
        if prev_gw not in gw_transactions:
            num_free_transfers = 2
        elif gw_transactions[prev_gw] >= 2:
            num_free_transfers = 1

    return num_free_transfers


def get_players_for_gameweek(
    gameweek: int,
    fpl_team_id: int | None = None,
    fetcher: FPLDataFetcher | None = None,
) -> list[Player]:
    """The players an entry had in a gameweek, from the FPL API."""
    fetcher = fetcher if fetcher is not None else get_fetcher()
    if not fpl_team_id:
        fpl_team_id = get_fetcher().FPL_TEAM_ID

    player_data = fetcher.get_fpl_team_data(gameweek, fpl_team_id)["picks"]
    player_api_id_list = [p["element"] for p in player_data]
    players: list[Player] = []
    for api_id in player_api_id_list:
        player = get_player_from_api_id(api_id)
        if player is None:
            logger.warning("Unable to find player with fpl_api_id %s", api_id)
            continue
        players.append(player)
    return players


def free_hit_used_in_gameweek(
    gameweek: int, fpl_team_id: int | None = None, fetcher: FPLDataFetcher | None = None
) -> int:
    """
    Whether the entry played its free hit in a gameweek, as 0 or 1.

    An int because that is how the transactions table records it.
    """
    fetcher = fetcher if fetcher is not None else get_fetcher(fpl_team_id)
    if fpl_team_id is None:
        fpl_team_id = fetcher.FPL_TEAM_ID
    fpl_team_data = fetcher.get_fpl_team_data(gameweek, fpl_team_id)
    return int(bool(fpl_team_data) and fpl_team_data.get("active_chip") == "freehit")
