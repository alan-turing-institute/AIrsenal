#!/usr/bin/env python

"""
simple script, check whether recent matches have been played since
the last entries in the DB.
"""

import os
import sys



from .fill_result_table import  fill_results_from_api
from .fill_playerscore_table import fill_playerscores_from_api
from .fill_transaction_table import add_transaction
from ..framework.utils import *
from ..framework.schema import session_scope

def main():
    season="1819"
    with session_scope() as session:

        last_in_db = get_last_gameweek_in_db(season, session)
        last_finished = get_last_finished_gameweek()
        if last_finished > last_in_db:
        ## need to update
            next_gw = get_next_gameweek()
            fill_results_from_api(last_in_db + 1, next_gw, season, session)

            fill_playerscores_from_api(last_in_db + 1, next_gw)
        else:
            print("Matches and player-scores already up-to-date")
        ## now check transfers
        print("Checking team")
        db_players = sorted(get_current_players(season=season,dbsession=session))
        api_players = sorted(get_players_for_gameweek(last_finished))
        if db_players != api_players:
            players_out = list(set(db_players).difference(api_players))
            players_in = list(set(api_players).difference(db_players))
            for p in players_out:
                add_transaction(p, last_finished, -1, season, session,
                                os.path.join(os.path.dirname(__file__), "../data/transactions.csv"))
            for p in players_in:
                add_transaction(p, last_finished, 1, season, session,
                                os.path.join(os.path.dirname(__file__), "../data/transactions.csv"))
        else:
            print("Team is up-to-date")
