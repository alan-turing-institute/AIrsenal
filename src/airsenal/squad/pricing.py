"""What a player in the squad would sell for."""

from sqlalchemy.orm import Session

from airsenal.core.logging import get_logger
from airsenal.db.queries.gameweeks import next_gameweek
from airsenal.db.queries.players import get_player
from airsenal.db.session import get_session
from airsenal.game.season import CURRENT_SEASON
from airsenal.remote.fpl_api import FPLDataFetcher, get_fetcher
from airsenal.squad.player import SquadPlayer

logger = get_logger(__name__)


def sell_price(
    player: SquadPlayer,
    *,
    use_api: bool = False,
    gameweek: int | None = None,
    season: str,
    fetcher: FPLDataFetcher | None = None,
    dbsession: Session | None = None,
) -> int:
    """Get sale price for a player in the squad, for the given gameweek.

    FPL gives back half of any rise in a player's price since we bought them,
    rounded down, which is the arithmetic at the end.
    """
    fetcher = fetcher if fetcher is not None else get_fetcher()
    gameweek = next_gameweek() if gameweek is None else gameweek
    dbsession = dbsession if dbsession is not None else get_session()
    player_id = player.player_id

    price_now = None
    player_db = get_player(player_id, dbsession=dbsession)
    if (
        use_api
        and season == CURRENT_SEASON
        and gameweek >= next_gameweek()
        and player_db is not None
        and player_db.fpl_api_id is not None
    ):
        api_id = player_db.fpl_api_id
        # first try getting the actual sale price from a logged in API
        selling_price = selling_price_from_api(api_id, player, fetcher=fetcher)
        if selling_price is not None:
            return selling_price
        # no selling price to be had, so use the player's current price
        try:
            price_now = fetcher.get_player_summary_data()[api_id]["now_cost"]
        except Exception:
            logger.warning(
                "Failed to get current price of %s from API. "
                "Will attempt to use latest price in DB instead.",
                player,
                exc_info=True,
            )

    # retrieve how much we originally bought the player for from db
    price_bought = player.purchase_price

    # get player's current price from db if the API wasn't used
    if not price_now and player_db:
        price_now = player_db.price(gameweek, season)

    # if all else fails just use the purchase price as the sale price for the player
    if not price_now:
        logger.warning(
            "Using purchase price as sale price for %s, %s",
            player.player_id,
            player,
        )
        price_now = price_bought

    if price_now > price_bought:
        return (price_now + price_bought) // 2
    return price_now


def selling_price_from_api(
    api_id: int,
    player: SquadPlayer,
    fetcher: FPLDataFetcher | None = None,
) -> int | None:
    """
    What the FPL API says this player would sell for, or None if it cannot say.

    A selling price exists only for a player the entry actually owns.
    """
    fetcher = fetcher if fetcher is not None else get_fetcher()
    try:
        picks = fetcher.get_current_picks()
    except Exception:
        logger.warning(
            "Failed to get the current picks from the FPL API to price %s. "
            "Will estimate based on the player's current price instead",
            player,
            exc_info=True,
        )
        return None

    if api_id not in picks:
        logger.debug(
            "%s is not in the FPL team's current picks, so the API has no sale "
            "price for them; using their current price instead",
            player,
        )
        return None

    try:
        return int(picks[api_id]["selling_price"])
    except (KeyError, TypeError, ValueError):
        logger.warning(
            "The FPL API returned no usable selling price for %s. "
            "Will estimate based on the player's current price instead",
            player,
            exc_info=True,
        )
        return None
