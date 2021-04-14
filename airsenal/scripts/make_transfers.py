"""
Script to apply recommended transfers 

Ref:
https://github.com/sk82jack/PSFPL/blob/master/PSFPL/Public/Invoke-FplTransfer.ps1
https://www.reddit.com/r/FantasyPL/comments/b4d6gv/fantasy_api_for_transfers/
"""
import argparse

from airsenal.framework.schema import TransferSuggestion
from airsenal.framework.utils import session, get_player_name
from airsenal.scripts.get_transfer_suggestions import get_transfer_suggestions, build_strategy_string


"""
steps:
1) From the scripts linked to above send a request to the server. 
2) Check for edge-cases
3) write a test. 
"""

def check_proceed():
    proceed = input("Apply Transfers? There is no turning back! (yes/no)")
    if proceed == "yes":
        print("Applying Transfers...")
        return True
    else: 
        return False

def get_gw_transfers():
    
    rows = get_transfer_suggestions(session, TransferSuggestion)
    current_gw = rows[0].gameweek
    players_out, players_in = [],[]
    line = "=" * 20
    output = "\nTransfers to apply \n" + line +"\n"
    for row in rows:
        if row.gameweek == current_gw:
            if row.in_or_out < 0: 
                players_out.append(row.player_id)
                output += "OUT "
            else:
                players_in.append(row.player_id)
                output += "IN "
    
            output += get_player_name(row.player_id) + "\n"
    
    output += line + "\n"
    print(output)

    return([players_in, players_out])

    
def apply_transfers(player_ids):
    """ post api request """
    pass


def main():
    parser = argparse.ArgumentParser(description="make transfers")

    parser.add_argument(
        "--players_out", help="list of players out", type=list, default=[]
    )
    parser.add_argument(
        "--players_in", help="list of players in", type=list, default=[]
    )   
 
    args = parser.parse_args()

    apply_transfers(
        args.players_out,
        args.players_in,
    )

if __name__ == "__main__":
    transfer_player_ids = get_gw_transfers()
    if check_proceed():
        apply_transfers(transfer_player_ids)
        