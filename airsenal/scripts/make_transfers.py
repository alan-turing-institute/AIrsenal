"""
Script to apply recommended transfers 

Ref:
https://github.com/sk82jack/PSFPL/blob/master/PSFPL/Public/Invoke-FplTransfer.ps1
https://www.reddit.com/r/FantasyPL/comments/b4d6gv/fantasy_api_for_transfers/
"""
import argparse
from prettytable import PrettyTable

from airsenal.framework.schema import TransferSuggestion
from airsenal.framework.utils import session, get_player_name, get_sell_price_for_player, get_bank
from airsenal.scripts.get_transfer_suggestions import get_transfer_suggestions, build_strategy_string

from airsenal.framework.data_fetcher import FPLDataFetcher

"""
{
	"confirmed": false,
	"entry": <entry>,
	"event": <next_gameweek>,
	"transfers": [
		{
		"element_in": <player_id>,
		"element_out": <player_id>,
		"purchase_price": <element_in_price>,
		"selling_price": <element_out_selling_price>
		}
	],
	"wildcard": false,
	"freehit": false
}
"""

"""
TODO:
- print bank balance
- confirm points loss
- From the scripts linked to above send a request to the server. 
- implement token use
- Check for edge-cases
- write a test. 
"""

def check_proceed():
    proceed = input("Apply Transfers? There is no turning back! (yes/no)")
    if proceed == "yes":
        print("Applying Transfers...")
        return True
    else: 
        return False

def price_transfers(transfer_player_ids, fetcher, current_gw):

    team_id = fetcher.FPL_TEAM_ID
    header = f"\nTransfers to apply for fpl_team_id: {team_id} for gameweek: {current_gw}"
    line = "=" * len(header)
    print(f"{header} \n {line} \n")

    bank = get_bank(fpl_team_id=team_id)
    print(f"Bank Balance Before transfers is: £{bank/10}")
    t = PrettyTable(['Status','Name','Price'])

    transfers = (list(zip(*transfer_player_ids))) #[(out,in),(out,in)]
    
    #[[[out, price], [in, price]],[[out,price],[in,price]]]
    priced_transfers = [[[t[0],get_sell_price_for_player(t[0], gameweek=current_gw, fpl_team_id=team_id)],
                        [t[1], fetcher.get_player_summary_data()[t[1]]["now_cost"]]]
                        for t in transfers]

    for transfer in priced_transfers:
        t.add_row(['OUT',get_player_name(transfer[0][0]),f"£{transfer[0][1]/10}"])
        t.add_row(['IN',get_player_name(transfer[1][0]),f"£{transfer[1][1]/10}"])

    print(t)

    print(f"Bank Balance After Transfers is: TODO")
    print(f"Points Cost of Transfers: TODO")
    return(price_transfers)


def get_gw_transfer_suggestions(fpl_team_id=None):
    
    ## gets the transfer suggestions for the latest optimization run, regardless of fpl_team_id
    rows = get_transfer_suggestions(session, TransferSuggestion)
    print(rows[0])
    if fpl_team_id and fpl_team_id != rows[0].fpl_team_id: 
        raise Exception(f'Team ID passed is {fpl_team_id}, but transfer suggestions are for team ID {rows[0].fpl_team_id}') 
    else:
        fpl_team_id = rows[0].fpl_team_id
    current_gw = rows[0].gameweek
    players_out, players_in = [],[]

    for row in rows:
        if row.gameweek == current_gw:
            print(row.in_or_out)
            if row.in_or_out < 0: 
                players_out.append(row.player_id)
            else:
                players_in.append(row.player_id) 

    return([players_out, players_in], fpl_team_id, current_gw)
    
    
def apply_transfers(player_ids):
    """ post api request """
    raise NotImplementedError


if __name__ == "__main__":
    
    transfer_player_ids, fpl_team_id, current_gw = get_gw_transfer_suggestions()
    fetcher = FPLDataFetcher(fpl_team_id)

    price_transfers(transfer_player_ids, fetcher, current_gw)
   

    if check_proceed():
        apply_transfers(transfer_player_ids)
        