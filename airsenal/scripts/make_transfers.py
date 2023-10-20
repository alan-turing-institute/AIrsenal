"""
Script to apply recommended transfers from the current transfer suggestion table.

Ref:
https://github.com/sk82jack/PSFPL/blob/master/PSFPL/Public/Invoke-FplTransfer.ps1
https://www.reddit.com/r/FantasyPL/comments/b4d6gv/fantasy_api_for_transfers/
https://fpl.readthedocs.io/en/latest/_modules/fpl/models/user.html#User.transfer
"""
import argparse
from typing import List, Optional, Tuple

from prettytable import PrettyTable

from airsenal.framework.data_fetcher import FPLDataFetcher
from airsenal.framework.optimization_utils import get_starting_squad
from airsenal.framework.utils import (
    CURRENT_SEASON,
    NEXT_GAMEWEEK,
    get_bank,
    get_player,
    get_player_from_api_id,
)
from airsenal.framework.utils import session as dbsession
from airsenal.scripts.get_transfer_suggestions import get_transfer_suggestions
from airsenal.scripts.set_lineup import set_lineup

"""
TODO:
- confirm points loss
- write a test.
"""


def check_proceed(num_transfers: int = 0) -> bool:
    proceed = input("Apply Transfers? There is no turning back! (yes/no)")
    if proceed != "yes":
        return False
    if num_transfers > 2:
        proceed = input(
            "Note that this script doesn't currently apply wildcard/free-hit chips.\n"
            "These transfers will result in points hit unless you play one of those "
            "chips via the website.  Are you sure you wish to proceed? (yes/no) "
        )
        if proceed != "yes":
            return False
    print("Applying Transfers...")
    return True


def deduct_transfer_price(pre_bank: float, priced_transfers: List[dict]) -> float:
    gain = [
        transfer["selling_price"] - transfer["purchase_price"]
        for transfer in priced_transfers
    ]
    return pre_bank + sum(gain)


def print_output(
    team_id: int,
    current_gw: int,
    priced_transfers: List[dict],
    pre_bank: Optional[float] = None,
    post_bank: Optional[float] = None,
    points_cost: str = "TODO",
) -> None:
    print("\n")
    header = f"Transfers to apply for fpl_team_id: {team_id} for gameweek: {current_gw}"
    line = "=" * len(header)
    print(f"{header} \n {line} \n")

    if pre_bank is not None:
        print(f"Bank Balance Before transfers is: £{pre_bank/10}")

    t = PrettyTable(["Status", "Name", "Price"])
    for transfer in priced_transfers:
        t.add_row(
            [
                "OUT",
                get_player_from_api_id(transfer["element_out"]),
                f"£{transfer['selling_price']/10}",
            ]
        )
        t.add_row(
            [
                "IN",
                get_player_from_api_id(transfer["element_in"]),
                f"£{transfer['purchase_price']/10}",
            ]
        )

    print(t)

    if post_bank is not None:
        print(f"Bank Balance After transfers is: £{post_bank/10}")
    # print(f"Points Cost of Transfers: {points_cost}")
    print("\n")


def get_sell_price(team_id: int, player_id: int, season: str = CURRENT_SEASON) -> float:
    squad = get_starting_squad(
        next_gw=NEXT_GAMEWEEK, season=season, fpl_team_id=team_id
    )
    for p in squad.players:
        if p.player_id == player_id:
            return squad.get_sell_price_for_player(p)


def get_gw_transfer_suggestions(
    fpl_team_id: Optional[int] = None,
) -> Optional[Tuple[List[list], int, int, str]]:
    # gets the transfer suggestions for the latest optimization run,
    # regardless of fpl_team_id
    rows = get_transfer_suggestions(
        dbsession,
        gameweek=NEXT_GAMEWEEK,
        season=CURRENT_SEASON,
        fpl_team_id=fpl_team_id,
    )
    if not rows:
        print(
            f"No transfer suggestions found for GW {NEXT_GAMEWEEK}, "
            f"{CURRENT_SEASON} season, FPL team id {fpl_team_id}"
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
    transfer_player_ids: List[list], fetcher: FPLDataFetcher
) -> List[dict]:
    """
    For most gameweeks, we get transfer suggestions from the db, including
    both players to be removed and added.
    """
    transfers = list(zip(*transfer_player_ids))  # [(out,in),(out,in)]
    priced_transfers = [
        [
            [t[0], get_sell_price(fetcher.FPL_TEAM_ID, t[0])],
            [
                t[1],
                fetcher.get_player_summary_data()[get_player(t[1]).fpl_api_id][
                    "now_cost"
                ],
            ],
        ]
        for t in transfers
    ]

    def to_dict(t):
        return {
            "element_out": get_player(t[0][0]).fpl_api_id,
            "selling_price": t[0][1],
            "element_in": get_player(t[1][0]).fpl_api_id,
            "purchase_price": t[1][1],
        }

    transfer_list = [to_dict(transfer) for transfer in priced_transfers]
    return transfer_list


def separate_transfers_in_or_out(transfer_list: List[dict]) -> Tuple[list, list]:
    """
    Given a list of dicts with keys
    "element_in", "purchase_price", "element_out", "selling_price",
    (such as what is returned by price_transfers),
    return two lists of dicts, one for transfers in and
    one for transfers out
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


def sort_by_position(transfer_list: List[dict]) -> list[dict]:
    """
    Takes a list of transfers e.g. [{"element_in": <FPL_API_ID>, "purchase_price": x}]
    and returns the same list ordered by DEF, FWD, GK, MID (i.e. alphabetical)
    to ensure that when we send a big list to the transfer API,
    we always replace like-with-like.

    Note that it is the FPL API ID used here, NOT the player_id.
    """

    def _get_position(api_id):
        return get_player_from_api_id(api_id).position(CURRENT_SEASON)

    # key to the dict could be either 'element_in' or 'element_out'.
    id_key = None
    for k, v in transfer_list[0].items():
        if "element" in k:
            id_key = k
            break
    if not id_key:
        raise RuntimeError(
            """
            sort_by_position expected a list of dicts,
            containing key 'element_in' or 'element_out'
            """
        )
    # now sort by position of the element_in/out player
    transfer_list = sorted(transfer_list, key=lambda k: _get_position(k[id_key]))
    return transfer_list


def remove_duplicates(transfers_in: List[int], transfers_out: List[int]) -> Tuple:
    """
    If we are replacing lots of players (e.g. new team), need to make sure there
    are no duplicates - can't add a player if we already have them.
    """
    t_in = [t["element_in"] for t in transfers_in]
    t_out = [t["element_out"] for t in transfers_out]
    dupes = list(set(t_in) & set(t_out))
    transfers_in = [t for t in transfers_in if not t["element_in"] in dupes]
    transfers_out = [t for t in transfers_out if not t["element_out"] in dupes]
    return transfers_in, transfers_out


def build_init_priced_transfers(
    fetcher: FPLDataFetcher, fpl_team_id: Optional[int] = None
) -> List[dict]:
    """
    Before gameweek 1, there won't be any 'sell' transfer suggestions in the db.
    We can instead query the API for our current 'picks' (requires login).
    """
    if not fpl_team_id:
        if (not fetcher.FPL_TEAM_ID) or fetcher.FPL_TEAM_ID == "MISSING_ID":
            fpl_team_id = int(input("Please enter FPL team ID: "))
        else:
            fpl_team_id = fetcher.FPL_TEAM_ID

    current_squad = fetcher.get_current_picks(fpl_team_id)
    transfers_out = [
        {"element_out": el["element"], "selling_price": el["selling_price"]}
        for el in current_squad
    ]
    transfer_in_suggestions = get_transfer_suggestions(dbsession)
    if len(transfers_out) != len(transfer_in_suggestions):
        raise RuntimeError(
            "Number of transfers in and out don't match: "
            f"{len(transfer_in_suggestions)} {len(transfers_out)}"
        )
    transfers_in = []
    for t in transfer_in_suggestions:
        api_id = get_player(t.player_id).fpl_api_id
        price = fetcher.get_player_summary_data()[api_id]["now_cost"]
        transfers_in.append({"element_in": api_id, "purchase_price": price})
    # remove duplicates - can't add a player we already have
    transfers_in, transfers_out = remove_duplicates(transfers_in, transfers_out)
    # re-order both lists so they go DEF, FWD, GK, MID
    transfers_in = sort_by_position(transfers_in)
    transfers_out = sort_by_position(transfers_out)
    transfer_list = [
        {**transfers_in[i], **transfers_out[i]} for i in range(len(transfers_in))
    ]
    return transfer_list


def build_transfer_payload(
    priced_transfers: List[dict],
    current_gw: int,
    fetcher: FPLDataFetcher,
    chip_played: str,
) -> dict:
    transfer_payload = {
        "confirmed": False,
        "entry": fetcher.FPL_TEAM_ID,
        "event": current_gw,
        "transfers": priced_transfers,
        "wildcard": False,
        "freehit": False,
    }
    if chip_played:
        transfer_payload[chip_played.replace("_", "")] = True

    print(transfer_payload)
    return transfer_payload


def make_transfers(
    fpl_team_id: Optional[int] = None, skip_check: bool = False
) -> Optional[bool]:
    suggestions = get_gw_transfer_suggestions(fpl_team_id)
    if not suggestions:
        return None
    transfer_player_ids, team_id, current_gw, chip_played = suggestions

    fetcher = FPLDataFetcher(team_id)
    if len(transfer_player_ids[0]) == 0:
        # no players to remove in DB - initial team?
        print("Making transfer list for starting team")
        priced_transfers = build_init_priced_transfers(fetcher, team_id)
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

    if skip_check or check_proceed(len(priced_transfers)):
        transfer_req = build_transfer_payload(
            priced_transfers, current_gw, fetcher, chip_played
        )
        fetcher.post_transfers(transfer_req)
    else:
        print("Not applying transfers.  Can still choose starting 11 and captain.")
        return False
    return True


def main():
    parser = argparse.ArgumentParser("Make transfers via the FPL API")
    parser.add_argument("--fpl_team_id", help="FPL team ID", type=int)
    parser.add_argument("--confirm", help="skip confirmation step", action="store_true")

    args = parser.parse_args()
    confirm = args.confirm or False
    try:
        make_transfers(args.fpl_team_id, confirm)
        set_lineup(args.fpl_team_id, skip_check=confirm)
    except Exception as e:
        raise Exception(
            "Something went wrong when making transfers. Check your team and make "
            "transfers and lineup changes manually on the web-site. If the problem "
            "persists, let us know on GitHub."
        ) from e


if __name__ == "__main__":
    main()
