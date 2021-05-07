"""
Script to apply recommended squad changes after transfers are made

"""

import requests
import json
from airsenal.framework.schema import TransferSuggestion
from airsenal.framework.optimization_utils import get_starting_squad
from airsenal.framework.utils import (
    get_latest_prediction_tag,
)
from airsenal.scripts.make_transfers import (
    get_gw_transfer_suggestions,
    price_transfers
)
from airsenal.framework.data_fetcher import FPLDataFetcher

"""
TODO:
- build payload by doing:
   - figuring out the indices order of players
   - sticking that in a dictionary
- send to api, using https://fpl.readthedocs.io/en/latest/_modules/fpl/models/user.html?highlight=lineup#
"""

"""
        new_lineup = [{
            "element": player["element"],
            "position": player["position"],
            "is_captain": player[is_c],
            "is_vice_captain": player[is_vc]
        } for player in lineup]

"""

def build_lineup_payload(squad):

    print(squad.players[0])


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

    build_lineup_payload(squad)


if __name__ == "__main__":

    main()
