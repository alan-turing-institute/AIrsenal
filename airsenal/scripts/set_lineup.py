"""
Script to apply recommended squad changes after transfers are made

"""

import requests
import json
from airsenal.framework.schema import TransferSuggestion
from airsenal.framework.optimization_utils import get_starting_squad
from airsenal.framework.utils import (
    get_latest_prediction_tag,
    get_player
)
from airsenal.scripts.make_transfers import (
    get_gw_transfer_suggestions,
    price_transfers,
    login
)
from airsenal.framework.data_fetcher import FPLDataFetcher

"""
TODO:
- apply chips
- send to api, using https://fpl.readthedocs.io/en/latest/_modules/fpl/models/user.html?highlight=lineup#
- incorporate into pipeline
"""

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
        return {"element": get_player(player.player_id).fpl_api_id,
                "position":pos_int,
                "is_captain":player.is_captain,
                "is_vice_captain":player.is_vice_captain
        }

    payload = []
    # payload for starting lineup
    lineup = [p for p in squad.players if p.is_starting]
    position_integer = 1
    for position_category in ["GK", "DEF", "MID", "FWD"]:
        for p in lineup:
            if p.position == position_category:
                payload.append(to_dict(p,position_integer))
                position_integer += 1
    
    
    sub_gk = [p for p in squad.players if not p.is_starting and p.position == "GK"][0]
    payload.append(to_dict(sub_gk,12))

    available_sub_positions = list(range(4))
    available_sub_positions.remove(sub_gk.sub_position)
    subs_outfield = [p for p in squad.players if not p.is_starting and p.position != "GK"]
    for s in subs_outfield:
        payload.append(
            to_dict(s, 13+available_sub_positions.index(s.sub_position))
            )        
             
    return payload

def post_lineup(payload, fetcher):

        session = requests.session()

        session = login(session, fetcher)

        payload = json.dumps({"chip": None, "picks": payload})
        headers = {
            "Content-Type": "application/json; charset=UTF-8",
            "X-Requested-With": "XMLHttpRequest",
            "Referer": "https://fantasy.premierleague.com/a/team/my",
        }

        team_url = f"https://fantasy.premierleague.com/api/my-team/{fetcher.FPL_TEAM_ID}/"

        resp = session.post(
            team_url, data=payload, headers=headers
        )
        if resp.status_code == 200:
            print("SUCCESS....lineup made!")
        else:
            print("Lineup changes not made due to unknown error")
            print(f"Response status code: {resp.status_code}")
            print(f"Response text: {resp.text}")
     

def make_squad_transfers(squad, priced_transfers):

    for t in priced_transfers:
       squad.remove_player(t[0][0], price = t[0][1])
       squad.add_player(t[1][0], price = t[1][1])


def main(fpl_team_id=None):

    squad = get_starting_squad(fpl_team_id)
    transfer_player_ids, team_id, gw, chip_played = get_gw_transfer_suggestions(fpl_team_id)
    
    fetcher = FPLDataFetcher(team_id)
    priced_transfers = price_transfers(transfer_player_ids, fetcher, gw)

    
    make_squad_transfers(squad, priced_transfers)
    squad.optimize_lineup(gw, get_latest_prediction_tag())


    if check_proceed(squad):
        payload = build_lineup_payload(squad)
        post_lineup(payload, fetcher)
    

if __name__ == "__main__":

    main(fpl_team_id=8149831)
