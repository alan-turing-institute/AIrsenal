"""
Applying the recommended transfers from the transfer suggestion table.

Ref:
https://github.com/sk82jack/PSFPL/blob/master/PSFPL/Public/Invoke-FplTransfer.ps1
https://www.reddit.com/r/FantasyPL/comments/b4d6gv/fantasy_api_for_transfers/
https://fpl.readthedocs.io/en/latest/_modules/fpl/models/user.html#User.transfer
"""

from typing import Any

from airsenal.core.console import confirm, console, table
from airsenal.core.logging import get_logger
from airsenal.db.queries.gameweeks import next_gameweek
from airsenal.db.queries.players import get_player, get_player_from_api_id
from airsenal.db.queries.predictions import get_transfer_suggestions
from airsenal.db.session import get_session
from airsenal.game.enums import Chip
from airsenal.game.season import CURRENT_SEASON
from airsenal.remote.fpl_api import FPLDataFetcher, get_fetcher
from airsenal.squad.history import get_starting_squad
from airsenal.squad.state import get_bank

logger = get_logger(__name__)


def check_proceed(num_transfers: int = 0) -> bool:
    """
    Ask before posting transfers to the real FPL entry.

    Through `confirm` rather than a bare `input()` so a test can stub one
    function, and with `default=False` because this is not reversible.
    """
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


def deduct_transfer_price(pre_bank: int, priced_transfers: list[dict[str, int]]) -> int:
    gain = [
        transfer["selling_price"] - transfer["purchase_price"]
        for transfer in priced_transfers
    ]
    return pre_bank + sum(gain)


def print_output(
    team_id: int,
    current_gw: int,
    priced_transfers: list[dict[str, int]],
    pre_bank: int | None = None,
    post_bank: int | None = None,
) -> None:
    console.print()
    header = f"Transfers to apply for fpl_team_id: {team_id} for gameweek: {current_gw}"
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


def get_sell_price(team_id: int, player_id: int, season: str = CURRENT_SEASON) -> int:
    squad = get_starting_squad(
        next_gw=next_gameweek(), season=season, fpl_team_id=team_id
    )
    for p in squad.players:
        if p.player_id == player_id:
            return squad.get_sell_price_for_player(p)

    msg = f"Player {player_id} not found in FPL team {team_id}"
    raise ValueError(msg)


def get_gw_transfer_suggestions(
    fpl_team_id: int | None = None,
) -> tuple[list[list[int]], int, int, str | None] | None:
    # the latest optimization run for this entry; without an fpl_team_id, for
    # whichever entry ran last
    rows = get_transfer_suggestions(
        gameweek=next_gameweek(),
        season=CURRENT_SEASON,
        fpl_team_id=fpl_team_id,
        dbsession=get_session(),
    )
    if not rows:
        logger.warning(
            "No transfer suggestions found for GW %s, %s season, FPL team id %s",
            next_gameweek(),
            CURRENT_SEASON,
            fpl_team_id,
        )
        return None

    if fpl_team_id is None:
        fpl_team_id = rows[0].fpl_team_id
    current_gw, chip = rows[0].gameweek, rows[0].chip_played
    players_out, players_in = [], []

    for row in rows:
        if row.gameweek == current_gw:
            if row.in_or_out < 0:
                players_out.append(row.player_id)
            else:
                players_in.append(row.player_id)
    return [players_out, players_in], fpl_team_id, current_gw, chip


def price_transfers(
    transfer_player_ids: list[list[int]], fetcher: FPLDataFetcher
) -> list[dict[str, int]]:
    """Pair up players out with players in, and price each pair for the API."""
    transfers = list(zip(*transfer_player_ids, strict=False))  # [(out,in),(out,in)]
    if fetcher.FPL_TEAM_ID is None:
        msg = "FPL team ID not set. Cannot price transfers."
        raise RuntimeError(msg)
    priced_transfers: list[list[list[int]]] = []
    for t in transfers:
        player = get_player(t[1])
        if player is None:
            msg = f"Player with ID {t[1]} not found"
            raise ValueError(msg)
        if player.fpl_api_id is None:
            msg = f"Player {player} has no FPL API ID"
            raise ValueError(msg)
        priced_transfers.append(
            [
                [t[0], get_sell_price(fetcher.FPL_TEAM_ID, t[0])],
                [
                    t[1],
                    int(
                        fetcher.get_player_summary_data()[player.fpl_api_id]["now_cost"]
                    ),
                ],
            ]
        )

    def to_dict(t: list[list[int]]) -> dict[str, int]:
        p_out = get_player(t[0][0])
        p_in = get_player(t[1][0])
        if not p_out or not p_in:
            msg = f"Player not found for transfer: {t}"
            raise ValueError(msg)
        if p_out.fpl_api_id is None or p_in.fpl_api_id is None:
            msg = f"Player without an FPL API ID in transfer: {t}"
            raise ValueError(msg)
        return {
            "element_out": p_out.fpl_api_id,
            "selling_price": t[0][1],
            "element_in": p_in.fpl_api_id,
            "purchase_price": t[1][1],
        }

    return [to_dict(transfer) for transfer in priced_transfers]


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


def sort_by_position(transfer_list: list[dict[str, int]]) -> list[dict[str, int]]:
    """
    Order transfers by position - DEF, FWD, GK, MID, i.e. alphabetically.

    Sending a long list to the transfer API replaces like with like positionally,
    so both halves have to be in the same order. The ids here are FPL API ids,
    not this database's player_ids.
    """

    def _get_position(api_id: int) -> str:
        player = get_player_from_api_id(api_id)
        if player is None:
            msg = f"Player with API ID {api_id} not found"
            raise ValueError(msg)
        pos = player.position(CURRENT_SEASON)
        if pos is None:
            msg = f"Player {player} has no position for season {CURRENT_SEASON}"
            raise ValueError(msg)
        return pos

    # key to the dict could be either 'element_in' or 'element_out'.
    id_key = None
    for k, _v in transfer_list[0].items():
        if "element" in k:
            id_key = k
            break
    if not id_key:
        msg = """
            sort_by_position expected a list of dicts,
            containing key 'element_in' or 'element_out'
            """
        raise RuntimeError(msg)
    # now sort by position of the element_in/out player
    return sorted(transfer_list, key=lambda k: _get_position(k[id_key]))


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
    dupes = list(set(t_in) & set(t_out))
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
            # a library function that stops to read stdin cannot be called from
            # anything but a terminal, and cannot be tested at all
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
    # Narrowed to this entry and this season. Unfiltered, the latest suggestion
    # anywhere in the table won: a replay's from-scratch squad build is fifteen
    # "in" rows for a dummy entry in a past season, which passes the length check
    # below and would be bought for the real entry.
    transfer_in_suggestions = get_transfer_suggestions(
        season=CURRENT_SEASON, fpl_team_id=fpl_team_id, dbsession=get_session()
    )
    if len(transfers_out) != len(transfer_in_suggestions):
        msg = (
            "Number of transfers in and out don't match: "
            f"{len(transfer_in_suggestions)} {len(transfers_out)}"
        )
        raise RuntimeError(msg)
    transfers_in = []
    for t in transfer_in_suggestions:
        player = get_player(t.player_id)
        if player is None:
            msg = f"Player with ID {t.player_id} not found"
            raise ValueError(msg)
        api_id = player.fpl_api_id
        if api_id is None:
            msg = f"Player {player} has no FPL API ID"
            raise ValueError(msg)
        price = fetcher.get_player_summary_data()[api_id]["now_cost"]
        transfers_in.append({"element_in": api_id, "purchase_price": price})
    # remove duplicates - can't add a player we already have
    transfers_in, transfers_out = remove_duplicates(transfers_in, transfers_out)
    # re-order both lists so they go DEF, FWD, GK, MID
    transfers_in = sort_by_position(transfers_in)
    transfers_out = sort_by_position(transfers_out)
    return [{**transfers_in[i], **transfers_out[i]} for i in range(len(transfers_in))]


# Only the two squad chips are part of a transfer. Bench boost and triple captain
# are lineup chips, and the endpoint has no field for them - stripping the
# underscore off the chip name posted a `benchboost` key the API does not define.
# Keyed by `Chip`, which is a StrEnum, so a plain chip string looks up fine.
TRANSFER_CHIP_FIELDS: dict[str, str] = {
    Chip.WILDCARD: "wildcard",
    Chip.FREE_HIT: "freehit",
}


def build_transfer_payload(
    priced_transfers: list[dict[str, int]],
    current_gw: int,
    fetcher: FPLDataFetcher,
    chip_played: str | None,
) -> dict[str, Any]:
    transfer_payload = {
        "confirmed": False,
        "entry": fetcher.FPL_TEAM_ID,
        "event": current_gw,
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
        skip_check: Post without asking. Ignored under `dry_run`, which never
            posts and so has nothing to ask about.
        dry_run: Build and show the payload, post nothing. The only way to see
            exactly what would be sent without sending it.
    """
    suggestions = get_gw_transfer_suggestions(fpl_team_id)
    if not suggestions:
        return None
    transfer_player_ids, team_id, current_gw, chip_played = suggestions

    fetcher = get_fetcher(team_id)
    if len(transfer_player_ids[0]) == 0:
        # no players to remove in DB - initial team?
        logger.info("Making transfer list for starting team")
        priced_transfers = build_init_priced_transfers(
            fpl_team_id=team_id, fetcher=fetcher
        )
        pre_transfer_bank = None
        post_transfer_bank = None
    else:
        pre_transfer_bank = get_bank(fpl_team_id=team_id)
        priced_transfers = price_transfers(transfer_player_ids, fetcher)
        # sort transfers by position
        transfers_out, transfers_in = separate_transfers_in_or_out(priced_transfers)
        sorted_transfers_out = sort_by_position(transfers_out)
        sorted_transfers_in = sort_by_position(transfers_in)
        priced_transfers = [
            {**sorted_transfers_out[i], **sorted_transfers_in[i]}
            for i in range(len(sorted_transfers_out))
        ]
        post_transfer_bank = deduct_transfer_price(pre_transfer_bank, priced_transfers)

    print_output(
        team_id,
        current_gw,
        priced_transfers,
        pre_transfer_bank,
        post_transfer_bank,
    )

    transfer_req = build_transfer_payload(
        priced_transfers, current_gw, fetcher, chip_played
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
