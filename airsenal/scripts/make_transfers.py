"""
Script to apply recommended transfers from the current transfer suggestion table.

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

def deduct_transfer_price(pre_bank, priced_transfers):

    gain = [transfer[0][1] - transfer[1][1] for transfer in priced_transfers]
    return(pre_bank + sum(gain))


def print_output(team_id, current_gw, priced_transfers, pre_bank, post_bank, points_cost='TODO'):

    print("\n")
    header = f"Transfers to apply for fpl_team_id: {team_id} for gameweek: {current_gw}"
    line = "=" * len(header)
    print(f"{header} \n {line} \n")

    print(f"Bank Balance Before transfers is: £{pre_bank/10}")

    t = PrettyTable(['Status','Name','Price'])
    for transfer in priced_transfers:
        t.add_row(['OUT',get_player_name(transfer[0][0]),f"£{transfer[0][1]/10}"])
        t.add_row(['IN',get_player_name(transfer[1][0]),f"£{transfer[1][1]/10}"])

    print(t)

    print(f"Bank Balance After transfers is: £{post_bank/10}")
    print(f"Points Cost of Transfers: {points_cost}")
    print("\n")


def price_transfers(transfer_player_ids, fetcher, current_gw):

    transfers = (list(zip(*transfer_player_ids))) #[(out,in),(out,in)]
    
    #[[[out, price], [in, price]],[[out,price],[in,price]]]
    priced_transfers = [[[t[0],get_sell_price_for_player(t[0], gameweek=current_gw, fpl_team_id=fetcher.FPL_TEAM_ID)],
                        [t[1], fetcher.get_player_summary_data()[t[1]]["now_cost"]]]
                        for t in transfers]

    return(priced_transfers)


def get_gw_transfer_suggestions(fpl_team_id=None):
    
    ## gets the transfer suggestions for the latest optimization run, regardless of fpl_team_id
    rows = get_transfer_suggestions(session, TransferSuggestion)
    if fpl_team_id and fpl_team_id != rows[0].fpl_team_id: 
        raise Exception(f'Team ID passed is {fpl_team_id}, but transfer suggestions are for team ID {rows[0].fpl_team_id}') 
    else:
        fpl_team_id = rows[0].fpl_team_id
    current_gw = rows[0].gameweek
    players_out, players_in = [],[]

    for row in rows:
        if row.gameweek == current_gw:
            if row.in_or_out < 0: 
                players_out.append(row.player_id)
            else:
                players_in.append(row.player_id) 

    return([players_out, players_in], fpl_team_id, current_gw)
    
    
def apply_transfers(player_ids):
    """ post api request """
    raise NotImplementedError


def main():

    transfer_player_ids, fpl_team_id, current_gw = get_gw_transfer_suggestions()
    fetcher = FPLDataFetcher(fpl_team_id)

    pre_transfer_bank = get_bank(fpl_team_id=fpl_team_id)
    priced_transfers = price_transfers(transfer_player_ids, fetcher, current_gw)
    post_transfer_bank = deduct_transfer_price(pre_transfer_bank, priced_transfers)

    print_output(fpl_team_id,current_gw, priced_transfers, pre_transfer_bank, post_transfer_bank)
    

    if check_proceed():
        apply_transfers(transfer_player_ids)
    

if __name__ == "__main__":
    
    main()
        