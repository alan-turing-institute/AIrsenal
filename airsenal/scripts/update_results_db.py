#!/usr/bin/env python

"""
simple script, check whether recent matches have been played since
the last entries in the DB.
"""

import os
import sys

import argparse

from .fill_result_table import  fill_results_from_api
from .fill_playerscore_table import fill_playerscores_from_api
from .fill_transaction_table import update_team
from ..framework.utils import *
from ..framework.schema import session_scope

def main():

    parser = argparse.ArgumentParser(description="fill db tables with recent scores and transactions")
    parser.add_argument("--season",help="season, in format e.g. '1819'",default=CURRENT_SEASON)
    parser.add_argument("--tag",help="identifying tag", default="AIrsenal1819")
    args = parser.parse_args()

    season = args.season
    tag = args.tag

    with session_scope() as session:

        last_in_db = get_last_gameweek_in_db(season, session)
        last_finished = get_last_finished_gameweek()
        if last_finished > last_in_db:
        ## need to update
            next_gw = get_next_gameweek()
            fill_results_from_api(last_in_db + 1, next_gw, season, session)

            fill_playerscores_from_api(season, session, last_in_db + 1, next_gw)
        else:
            print("Matches and player-scores already up-to-date")
        ## now check transfers
        print("Checking team")
        db_players = sorted(get_current_players(season=season,dbsession=session))
        api_players = sorted(get_players_for_gameweek(last_finished))
        if db_players != api_players:
            update_team(season=season, session=session)
        else:
            print("Team is up-to-date")


if __name__ == "__main__":
    print(" ==== updating results and transactions === ")
    main()
