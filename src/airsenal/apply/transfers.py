"""
Applying the recommended transfers from the transfer suggestion table.

Ref:
https://github.com/sk82jack/PSFPL/blob/master/PSFPL/Public/Invoke-FplTransfer.ps1
https://www.reddit.com/r/FantasyPL/comments/b4d6gv/fantasy_api_for_transfers/
https://fpl.readthedocs.io/en/latest/_modules/fpl/models/user.html#User.transfer
"""

from typing import Any, NamedTuple

from airsenal.core.console import confirm, console, table
from airsenal.core.logging import get_logger
from airsenal.db.queries.gameweeks import next_gameweek
from airsenal.db.queries.players import (
    get_player_from_api_id,
    require_api_id,
    require_player,
    require_player_from_api_id,
)
from airsenal.db.queries.predictions import get_transfer_suggestions
from airsenal.db.session import get_session
from airsenal.game.enums import Chip
from airsenal.game.season import CURRENT_SEASON
from airsenal.remote.fpl_api import FPLDataFetcher, get_fetcher
from airsenal.squad.history import get_starting_squad
from airsenal.squad.squad import Squad
from airsenal.squad.state import get_bank

logger = get_logger(__name__)


def check_proceed(num_transfers: int = 0) -> bool:
    """Ask before posting transfers to the real FPL entry."""
    if not confirm("Apply transfers? There is no turning back!", default=False):
        return False
    if num_transfers > 2 and not confirm(
        "AIrsenal does not play the wildcard or free-hit chip for you, so these "
        "transfers will cost a points hit unless you play one on the website. "
        "Proceed?",
        default=False,
    ):
        return False
    console.print("Applying Transfers...")
    return True


def bank_after_transfers(pre_bank: int, priced_transfers: list[dict[str, int]]) -> int:
    """What is left in the bank once every player is sold and every one bought."""
    gain = [
        transfer["selling_price"] - transfer["purchase_price"]
        for transfer in priced_transfers
    ]
    return pre_bank + sum(gain)


def print_output(
    team_id: int,
    gameweek: int,
    priced_transfers: list[dict[str, int]],
    pre_bank: int | None = None,
    post_bank: int | None = None,
) -> None:
    console.print()
    header = f"Transfers to apply for fpl_team_id: {team_id} for gameweek: {gameweek}"
    line = "=" * len(header)
    console.print(f"{header}\n{line}")

    if pre_bank is not None:
        console.print(f"Bank Balance Before transfers is: £{pre_bank / 10}")

    transfer_table = table("Status", "Name", "Price")
    for transfer in priced_transfers:
        transfer_table.add_row(
            "OUT",
            str(get_player_from_api_id(transfer["element_out"])),
            f"£{transfer['selling_price'] / 10}",
        )
        transfer_table.add_row(
            "IN",
            str(get_player_from_api_id(transfer["element_in"])),
            f"£{transfer['purchase_price'] / 10}",
        )

    console.print(transfer_table)

    if post_bank is not None:
        console.print(f"Bank Balance After transfers is: £{post_bank / 10}")
    console.print()


def _entry_squad(team_id: int, season: str, fetcher: FPLDataFetcher) -> Squad:
    """The squad this entry holds going into the next gameweek."""
    return get_starting_squad(
        gameweek=next_gameweek(),
        season=season,
        fpl_team_id=team_id,
        fetcher=fetcher,
    )


def get_sell_price(
    team_id: int,
    player_id: int,
    season: str = CURRENT_SEASON,
    fetcher: FPLDataFetcher | None = None,
    *,
    squad: Squad | None = None,
) -> int:
    """
    What this entry can sell a player for, as the FPL API prices them.

    The transfer endpoint is given this figure. When the API cannot say, it is
    estimated from the transactions table instead, which is only right while the
    database is in step with the entry.

    Args:
        squad: The entry's squad, if the caller already has it.
    """
    fetcher = fetcher if fetcher is not None else get_fetcher(team_id)
    if squad is None:
        squad = _entry_squad(team_id, season, fetcher)
    for p in squad.players:
        if p.player_id == player_id:
            return squad.get_sell_price_for_player(p, use_api=True, fetcher=fetcher)

    msg = f"Player {player_id} not found in FPL team {team_id}"
    raise ValueError(msg)


class SuggestedTransfers(NamedTuple):
    """One gameweek's worth of suggestions, as the latest optimization run left them."""

    players_out: list[int]
    players_in: list[int]
    fpl_team_id: int
    gameweek: int
    chip_played: str | None


def get_suggested_transfers(
    fpl_team_id: int | None = None,
) -> SuggestedTransfers | None:
    """
    The next gameweek's suggestions, or None when the run left none to apply.

    Without an `fpl_team_id`, the suggestions belong to whichever entry ran last.
    """
    gameweek = next_gameweek()
    rows = get_transfer_suggestions(
        gameweek=gameweek,
        season=CURRENT_SEASON,
        fpl_team_id=fpl_team_id,
        dbsession=get_session(),
    )
    if not rows:
        logger.warning(
            "No transfer suggestions found for gameweek %s, %s season, FPL team id %s",
            gameweek,
            CURRENT_SEASON,
            fpl_team_id,
        )
        return None

    if fpl_team_id is None:
        fpl_team_id = rows[0].fpl_team_id
    chip = rows[0].chip_played
    players_out, players_in = [], []

    for row in rows:
        if row.in_or_out < 0:
            players_out.append(row.player_id)
        else:
            players_in.append(row.player_id)
    return SuggestedTransfers(players_out, players_in, fpl_team_id, gameweek, chip)


def price_transfers(
    players_out: list[int], players_in: list[int], fetcher: FPLDataFetcher
) -> list[dict[str, int]]:
    """
    Pair each player out with a player in, and price both sides for the API.

    A player is sold at the price the entry can get for them, which is not the
    market price: FPL gives back the purchase price plus half of any rise.
    """
    if fetcher.FPL_TEAM_ID is None:
        msg = "FPL team ID not set. Cannot price transfers."
        raise RuntimeError(msg)
    now_cost = fetcher.get_player_summary_data()
    squad = _entry_squad(fetcher.FPL_TEAM_ID, CURRENT_SEASON, fetcher)
    priced_transfers = []
    for player_id_out, player_id_in in zip(players_out, players_in, strict=True):
        api_id_in = require_api_id(require_player(player_id_in))
        priced_transfers.append(
            {
                "element_out": require_api_id(require_player(player_id_out)),
                "selling_price": get_sell_price(
                    fetcher.FPL_TEAM_ID, player_id_out, fetcher=fetcher, squad=squad
                ),
                "element_in": api_id_in,
                "purchase_price": int(now_cost[api_id_in]["now_cost"]),
            }
        )
    return priced_transfers


def separate_transfers_in_or_out(
    transfer_list: list[dict[str, int]],
) -> tuple[list[dict[str, int]], list[dict[str, int]]]:
    """
    Split `price_transfers` output into the transfers out and the transfers in.

    Each input dict carries all four of "element_in", "purchase_price",
    "element_out" and "selling_price"; the API wants the two halves separately.
    """
    transfers_out = [
        {"element_out": t["element_out"], "selling_price": t["selling_price"]}
        for t in transfer_list
    ]
    transfers_in = [
        {"element_in": t["element_in"], "purchase_price": t["purchase_price"]}
        for t in transfer_list
    ]
    return transfers_out, transfers_in


def _position_of(api_id: int) -> str:
    """The position of the player with this FPL API id, not this database's id."""
    player = require_player_from_api_id(api_id)
    position = player.position(CURRENT_SEASON)
    if position is None:
        msg = f"Player {player} has no position for season {CURRENT_SEASON}"
        raise ValueError(msg)
    return position


def pair_by_position(
    transfers_out: list[dict[str, int]], transfers_in: list[dict[str, int]]
) -> list[dict[str, int]]:
    """
    Order both halves by position, then pair them back into whole transfers.

    Sending a long list to the transfer API replaces like with like positionally,
    so a player out has to sit opposite a player in of the same position. Sorting
    on the position string does that: DEF, FWD, GK, MID, i.e. alphabetically.
    """
    return [
        {**out, **in_}
        for out, in_ in zip(
            sorted(transfers_out, key=lambda t: _position_of(t["element_out"])),
            sorted(transfers_in, key=lambda t: _position_of(t["element_in"])),
            strict=True,
        )
    ]


def sorted_by_position(
    priced_transfers: list[dict[str, int]],
) -> list[dict[str, int]]:
    """Re-pair priced transfers so each side lines up with the other by position."""
    transfers_out, transfers_in = separate_transfers_in_or_out(priced_transfers)
    return pair_by_position(transfers_out, transfers_in)


def remove_duplicates(
    transfers_in: list[dict[str, int]], transfers_out: list[dict[str, int]]
) -> tuple[list[dict[str, int]], list[dict[str, int]]]:
    """
    Drop any player appearing on both sides of the transfer list.

    Replacing most of a squad at once can otherwise ask the API to buy a player
    it is selling in the same request.
    """
    t_in = [t["element_in"] for t in transfers_in]
    t_out = [t["element_out"] for t in transfers_out]
    dupes = set(t_in) & set(t_out)
    transfers_in = [t for t in transfers_in if t["element_in"] not in dupes]
    transfers_out = [t for t in transfers_out if t["element_out"] not in dupes]
    return transfers_in, transfers_out


def build_init_priced_transfers(
    *, fpl_team_id: int | None = None, fetcher: FPLDataFetcher
) -> list[dict[str, int]]:
    """
    Price the transfers out from the API's current picks rather than the database.

    Before gameweek 1 there are no 'sell' suggestions in the database to price.
    Requires login.
    """
    if not fpl_team_id:
        if not fetcher.FPL_TEAM_ID:
            msg = (
                "No FPL team ID. Pass fpl_team_id, or set FPL_TEAM_ID with "
                "`airsenal env set FPL_TEAM_ID <id>`."
            )
            raise ValueError(msg)
        fpl_team_id = fetcher.FPL_TEAM_ID

    current_squad = fetcher.get_current_picks(fpl_team_id)
    transfers_out = [
        {"element_out": el["element"], "selling_price": el["selling_price"]}
        for el in current_squad.values()
    ]

    transfer_in_suggestions = get_transfer_suggestions(
        season=CURRENT_SEASON, fpl_team_id=fpl_team_id, dbsession=get_session()
    )
    if len(transfers_out) != len(transfer_in_suggestions):
        msg = (
            "Number of transfers in and out don't match: "
            f"{len(transfer_in_suggestions)} {len(transfers_out)}"
        )
        raise RuntimeError(msg)
    now_cost = fetcher.get_player_summary_data()
    transfers_in = []
    for t in transfer_in_suggestions:
        api_id = require_api_id(require_player(t.player_id))
        price = now_cost[api_id]["now_cost"]
        transfers_in.append({"element_in": api_id, "purchase_price": price})
    # remove duplicates - can't add a player we already have
    transfers_in, transfers_out = remove_duplicates(transfers_in, transfers_out)
    return pair_by_position(transfers_out, transfers_in)


# Only the two squad chips are part of a transfer. Bench boost and triple captain
# are lineup chips.
TRANSFER_CHIP_FIELDS: dict[str, str] = {
    Chip.WILDCARD: "wildcard",
    Chip.FREE_HIT: "freehit",
}


def build_transfer_payload(
    priced_transfers: list[dict[str, int]],
    chip_played: str | None,
    gameweek: int,
    fetcher: FPLDataFetcher,
) -> dict[str, Any]:
    """The body of a transfers request, with the chip flags the endpoint accepts."""
    transfer_payload = {
        "confirmed": False,
        "entry": fetcher.FPL_TEAM_ID,
        "event": gameweek,
        "transfers": priced_transfers,
        "wildcard": False,
        "freehit": False,
    }
    if chip_played:
        field = TRANSFER_CHIP_FIELDS.get(chip_played)
        if field is None:
            logger.info(
                "%s is not activated through the transfers endpoint - play it on "
                "the website.",
                chip_played,
            )
        else:
            transfer_payload[field] = True

    logger.debug("%s", transfer_payload)
    return transfer_payload


def make_transfers(
    fpl_team_id: int | None = None,
    skip_check: bool = False,
    dry_run: bool = False,
) -> bool | None:
    """
    Post the suggested transfers to the FPL entry.

    Returns None when there is nothing to apply, False when the user declined,
    and True when the transfers were posted - or, under `dry_run`, would have
    been.

    Args:
        skip_check: Post without asking. Ignored under `dry_run`.
        dry_run: Build and show the payload, post nothing.
    """
    suggested = get_suggested_transfers(fpl_team_id)
    if suggested is None:
        return None
    team_id = suggested.fpl_team_id

    fetcher = get_fetcher(team_id)
    if len(suggested.players_out) == 0:
        # no players to remove in DB - initial team?
        logger.info("Making transfer list for starting team")
        priced_transfers = build_init_priced_transfers(
            fpl_team_id=team_id, fetcher=fetcher
        )
        pre_transfer_bank = None
        post_transfer_bank = None
    else:
        pre_transfer_bank = get_bank(fpl_team_id=team_id)
        priced_transfers = sorted_by_position(
            price_transfers(suggested.players_out, suggested.players_in, fetcher)
        )
        post_transfer_bank = bank_after_transfers(pre_transfer_bank, priced_transfers)

    print_output(
        team_id,
        suggested.gameweek,
        priced_transfers,
        pre_transfer_bank,
        post_transfer_bank,
    )

    transfer_req = build_transfer_payload(
        priced_transfers, suggested.chip_played, suggested.gameweek, fetcher
    )
    if dry_run:
        console.print("[bold]Dry run: this is what would be posted[/bold]")
        console.print(transfer_req)
        return True
    if skip_check or check_proceed(len(priced_transfers)):
        fetcher.post_transfers(transfer_req)
    else:
        logger.info(
            "Not applying transfers.  Can still choose starting 11 and captain."
        )
        return False
    return True
