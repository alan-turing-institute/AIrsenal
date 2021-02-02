"""
Script to apply transfers, recommended or given

Ref:
https://github.com/sk82jack/PSFPL/blob/master/PSFPL/Public/Invoke-FplTransfer.ps1
https://www.reddit.com/r/FantasyPL/comments/b4d6gv/fantasy_api_for_transfers/
"""
import argparse

from airsenal.framework.schema import session_scope
from airsenal.framework.data_fetcher import API_HOME

""""
steps:
1) Figure out how the transfer suggestions get saved. 
2) Design a command line interface. 
3) Create a dummy team for testing. 
4) Have a 'are you sure?'
5) From the scripts linked to above send a request to the server. 


data held in: /tmp/data.db
interface into db:
sqlite3 /tmp/data.db
.tables

""""

"""

class TransferSuggestion(Base):
    __tablename__ = "transfer_suggestion"
    id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(Integer, nullable=False)
    in_or_out = Column(Integer, nullable=False)  # +1 for buy, -1 for sell
    gameweek = Column(Integer, nullable=False)
    points_gain = Column(Float, nullable=False)
    timestamp = Column(String(100), nullable=False)  # use this to group suggestions
    season = Column(String(100), nullable=False)


"""


def rerun_predictions(season, gw_start, gw_end, weeks_ahead=3, num_thread=4):
    """
    Run the predictions each week for gw_start to gw_end in chosen season.
    """
    with session_scope() as session:
        for gw in range(gw_start, gw_end + 1):
            print(
                "======== Running predictions for {} week {} ======".format(season, gw)
            )
            tag_prefix = season + "_" + str(gw) + "_"
            make_predictedscore_table(
                gw_range=range(gw, gw + weeks_ahead),
                season=season,
                num_thread=num_thread,
                tag_prefix=tag_prefix,
                dbsession=session,
            )

def main():
    parser = argparse.ArgumentParser(description="make transfers")

    parser.add_argument(
        "--gameweek_start", help="first gameweek to look at", type=int, default=1
    )
    parser.add_argument(
        "--gameweek_end", help="last gameweek to look at", type=int, default=38
    )
    parser.add_argument(
        "--weeks_ahead", help="how many weeks ahead to fill", type=int, default=3
    )
    parser.add_argument(
        "--season", help="season, in format e.g. '1819'", type=str, required=True
    )
    parser.add_argument(
        "--num_thread",
        help="number of threads to parallelise over",
        type=int,
        default=4,
    )
    args = parser.parse_args()

    rerun_predictions(
        args.season,
        args.gameweek_start,
        args.gameweek_end,
        args.weeks_ahead,
        args.num_thread,
    )
