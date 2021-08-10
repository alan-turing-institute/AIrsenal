"""
Script to apply recommended transfers from the current transfer suggestion table.

Ref:
https://github.com/sk82jack/PSFPL/blob/master/PSFPL/Public/Invoke-FplTransfer.ps1
https://www.reddit.com/r/FantasyPL/comments/b4d6gv/fantasy_api_for_transfers/
https://fpl.readthedocs.io/en/latest/_modules/fpl/models/user.html#User.transfer
"""
from prettytable import PrettyTable
import requests
import json
import getpass
from airsenal.framework.optimization_utils import get_starting_squad
from airsenal.framework.utils import (
    session as dbsession,
    get_player_name,
    get_bank,
    get_player,
    CURRENT_SEASON,
    get_player_from_api_id,
)
from airsenal.scripts.get_transfer_suggestions import get_transfer_suggestions
from airsenal.framework.data_fetcher import FPLDataFetcher

"""
TODO:
- confirm points loss
- write a test.
"""


def check_proceed():
    proceed = input("Apply Transfers? There is no turning back! (yes/no)")
    if proceed != "yes":
        return False

    print("Applying Transfers...")
    return True


def deduct_transfer_price(pre_bank, priced_transfers):

    gain = [transfer[0][1] - transfer[1][1] for transfer in priced_transfers]
    return pre_bank + sum(gain)


def print_output(
    team_id, current_gw, priced_transfers, pre_bank, post_bank, points_cost="TODO"
):

    print("\n")
    header = f"Transfers to apply for fpl_team_id: {team_id} for gameweek: {current_gw}"
    line = "=" * len(header)
    print(f"{header} \n {line} \n")

    print(f"Bank Balance Before transfers is: £{pre_bank/10}")

    t = PrettyTable(["Status", "Name", "Price"])
    for transfer in priced_transfers:
        t.add_row(["OUT", get_player_name(transfer[0][0]), f"£{transfer[0][1]/10}"])
        t.add_row(["IN", get_player_name(transfer[1][0]), f"£{transfer[1][1]/10}"])

    print(t)

    print(f"Bank Balance After transfers is: £{post_bank/10}")
    # print(f"Points Cost of Transfers: {points_cost}")
    print("\n")


def get_sell_price(team_id, player_id):

    squad = get_starting_squad(team_id)
    for p in squad.players:
        if p.player_id == player_id:
            return squad.get_sell_price_for_player(p)


def get_gw_transfer_suggestions(fpl_team_id=None):

    # gets the transfer suggestions for the latest optimization run,
    # regardless of fpl_team_id
    rows = get_transfer_suggestions(dbsession)
    if fpl_team_id and fpl_team_id != rows[0].fpl_team_id:
        raise Exception(
            f"Team ID passed is {fpl_team_id}, but transfer suggestions are for \
            team ID {rows[0].fpl_team_id}. We recommend re-running optimization."
        )
    else:
        fpl_team_id = rows[0].fpl_team_id
    current_gw, chip = rows[0].gameweek, rows[0].chip_played
    players_out, players_in = [], []

    for row in rows:
        if row.gameweek == current_gw:
            if row.in_or_out < 0:
                players_out.append(row.player_id)
            else:
                players_in.append(row.player_id)
    return ([players_out, players_in], fpl_team_id, current_gw, chip)


def price_transfers(transfer_player_ids, fetcher, current_gw):
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


def sort_by_position(transfer_list):
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


def remove_duplicates(transfers_in, transfers_out):
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


def build_init_priced_transfers(fetcher, fpl_team_id=None):
    """
    Before gameweek 1, there won't be any 'sell' transfer suggestions in the db.
    We can instead query the API for our current 'picks' (requires login).
    """
    if not fpl_team_id:
        if (not fetcher.FPL_TEAM_ID) or fetcher.FPL_TEAM_ID == "MISSING_ID":
            fpl_team_id = int(input("Please enter FPL team ID: "))
        else:
            fpl_team_id = fetcher.FPL_TEAM_ID

    current_squad = fetcher.get_current_squad_data(fpl_team_id)
    transfers_out = [
        {"element_out": el["element"], "selling_price": el["selling_price"]}
        for el in current_squad
    ]
    transfer_in_suggestions = get_transfer_suggestions(dbsession)
    if len(transfers_out) != len(transfer_in_suggestions):
        raise RuntimeError(
            "Number of transfers in and out don't match: {} {}".format(
                len(transfer_in_suggestions), len(transfers_out)
            )
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


def build_transfer_payload(priced_transfers, current_gw, fetcher, chip_played):

    transfer_payload = {
        "confirmed": False,
        "entry": fetcher.FPL_TEAM_ID,  # not sure what the entry should refer to?
        "event": current_gw,
        "transfers": priced_transfers,
        "wildcard": False,
        "freehit": False,
    }
    if chip_played:
        transfer_payload[chip_played.replace("_", "")] = True

    print(transfer_payload)

    return transfer_payload


def login(session, fetcher):

    if (
        (not fetcher.FPL_LOGIN)
        or (not fetcher.FPL_PASSWORD)
        or (fetcher.FPL_LOGIN == "MISSING_ID")
        or (fetcher.FPL_PASSWORD == "MISSING_ID")
    ):
        fetcher.FPL_LOGIN = input("Please enter FPL login: ")
        fetcher.FPL_PASSWORD = getpass.getpass("Please enter FPL password: ")

    # print("FPL credentials {} {}".format(fetcher.FPL_LOGIN, fetcher.FPL_PASSWORD))
    login_url = "https://users.premierleague.com/accounts/login/"
    headers = {
        "login": fetcher.FPL_LOGIN,
        "password": fetcher.FPL_PASSWORD,
        "app": "plfpl-web",
        "redirect_uri": "https://fantasy.premierleague.com/a/login",
    }
    session.post(login_url, data=headers)
    return session


def post_transfers(transfer_payload, fetcher):

    session = requests.session()

    session = login(session, fetcher)

    # adapted from https://github.com/amosbastian/fpl/blob/master/fpl/utils.py
    headers = {
        "Content-Type": "application/json; charset=UTF-8",
        "X-Requested-With": "XMLHttpRequest",
        "Referer": "https://fantasy.premierleague.com/a/squad/transfers",
    }

    transfer_url = "https://fantasy.premierleague.com/api/transfers/"

    resp = session.post(
        transfer_url, data=json.dumps(transfer_payload), headers=headers
    )
    if "non_form_errors" in resp:
        raise Exception(resp["non_form_errors"])
    elif resp.status_code == 200:
        print("SUCCESS....transfers made!")
    else:
        print("Transfers unsuccessful due to unknown error")
        print(f"Response status code: {resp.status_code}")
        print(f"Response text: {resp.text}")


def make_transfers(fpl_team_id=None):

    transfer_player_ids, team_id, current_gw, chip_played = get_gw_transfer_suggestions(
        fpl_team_id
    )
    fetcher = FPLDataFetcher(team_id)
    if len(transfer_player_ids[0]) == 0:
        # no players to remove in DB - initial team?
        print("Making transfer list for starting team")
        priced_transfers = build_init_priced_transfers(fetcher, team_id)
    else:

        pre_transfer_bank = get_bank(fpl_team_id=team_id)
        priced_transfers = price_transfers(transfer_player_ids, fetcher, current_gw)
        post_transfer_bank = deduct_transfer_price(pre_transfer_bank, priced_transfers)
        print_output(
            team_id, current_gw, priced_transfers, pre_transfer_bank, post_transfer_bank
        )

    if check_proceed():
        transfer_req = build_transfer_payload(
            priced_transfers, current_gw, fetcher, chip_played
        )
        post_transfers(transfer_req, fetcher)
    return True


if __name__ == "__main__":

    make_transfers()
