"""
Script to apply recommended squad changes after transfers are made

"""

from airsenal.framework.squad import Squad
import argparse
import requests
import json
from airsenal.framework.utils import (
    get_latest_prediction_tag,
    get_player,
    get_player_from_api_id,
    NEXT_GAMEWEEK,
)
from airsenal.scripts.make_transfers import (
    login,
)
from airsenal.framework.data_fetcher import FPLDataFetcher


def check_proceed(squad):

    print(squad)
    proceed = input("Apply changes to lineup? (yes/no) ")
    if proceed == "yes":
        print("Applying Changes...")
        return True
    else:
        return False


def build_lineup_payload(squad):
    def to_dict(player, pos_int):
        return {
            "element": get_player(player.player_id).fpl_api_id,
            "position": pos_int,
            "is_captain": player.is_captain,
            "is_vice_captain": player.is_vice_captain,
        }

    payload = []
    # payload for starting lineup
    lineup = [p for p in squad.players if p.is_starting]
    position_integer = 1
    for position_category in ["GK", "DEF", "MID", "FWD"]:
        for p in lineup:
            if p.position == position_category:
                payload.append(to_dict(p, position_integer))
                position_integer += 1

    sub_gk = [p for p in squad.players if not p.is_starting and p.position == "GK"][0]
    payload.append(to_dict(sub_gk, 12))

    available_sub_positions = list(range(4))
    available_sub_positions.remove(sub_gk.sub_position)
    subs_outfield = [
        p for p in squad.players if not p.is_starting and p.position != "GK"
    ]
    for s in subs_outfield:
        payload.append(to_dict(s, 13 + available_sub_positions.index(s.sub_position)))

    return payload


def get_lineup_from_payload(lineup):
    """
    inverse of build_lineup_payload. Returns a squad object from get_lineup

    lineup is a dictionary, with the entry "picks" being a list of dictionaries like:
    {"element":353,"position":1,"selling_price":55,"multiplier":1,"purchase_price":55,"is_captain":false,"is_vice_captain":false}
    """
    lineup = json.loads(lineup)
    s = Squad()
    for p in lineup["picks"]:
        player = get_player_from_api_id(p["element"])
        s.add_player(player, check_budget=False)

    if s.is_complete():
        return s
    else:
        raise RuntimeError("Squad incomplete")


def get_lineup(fetcher):
    """Retrieve up to date lineup from api"""

    req_session = requests.session()

    req_session = login(req_session, fetcher)

    headers = {
        "Content-Type": "application/json; charset=UTF-8",
        "X-Requested-With": "XMLHttpRequest",
        "Referer": "https://fantasy.premierleague.com/a/team/my",
    }

    team_url = f"https://fantasy.premierleague.com/api/my-team/{fetcher.FPL_TEAM_ID}/"

    resp = req_session.get(team_url, headers=headers)

    return resp.text


def post_lineup(payload, fetcher):

    req_session = requests.session()

    req_session = login(req_session, fetcher)

    payload = json.dumps({"chip": None, "picks": payload})
    headers = {
        "Content-Type": "application/json; charset=UTF-8",
        "X-Requested-With": "XMLHttpRequest",
        "Referer": "https://fantasy.premierleague.com/a/team/my",
    }

    team_url = f"https://fantasy.premierleague.com/api/my-team/{fetcher.FPL_TEAM_ID}/"

    resp = req_session.post(team_url, data=payload, headers=headers)
    if resp.status_code == 200:
        print("SUCCESS....lineup made!")
    else:
        print("Lineup changes not made due to unknown error")
        print(f"Response status code: {resp.status_code}")
        print(f"Response text: {resp.text}")


def make_squad_transfers(squad, priced_transfers):

    for t in priced_transfers:
        squad.remove_player(t[0][0], price=t[0][1])
        squad.add_player(t[1][0], price=t[1][1])


def set_lineup(fpl_team_id=None):

    """
    Retrieve the latest lineup and apply the latest prediction to it.

    Note that this assumes that the prediction has been ran recently.
    """
    print("fpl_team_id is {}".format(fpl_team_id))
    fetcher = FPLDataFetcher(fpl_team_id)
    print("Got fetcher {}".format(fetcher.FPL_TEAM_ID))
    picks = get_lineup(fetcher)
    print("Got picks {}".format(picks))
    squad = get_lineup_from_payload(picks)
    print("got squad: {}".format(squad))

    squad.optimize_lineup(NEXT_GAMEWEEK, get_latest_prediction_tag())

    if check_proceed(squad):
        payload = build_lineup_payload(squad)
        post_lineup(payload, fetcher)


def main():
    parser = argparse.ArgumentParser("Set the starting 11 and captain")
    parser.add_argument("--fpl_team_id", help="ID of the squad in FPL API")
    args = parser.parse_args()
    set_lineup(args.fpl_team_id)


if __name__ == "__main__":
    main()
