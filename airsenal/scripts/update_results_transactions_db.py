#!/usr/bin/env python

"""
simple script, check whether recent matches have been played since
the last entries in the DB, and update the transactions table with players
bought or sold.
"""

import os
import sys

import argparse

from .fill_player_attributes_table import fill_attributes_table_from_api
from .fill_result_table import fill_results_from_api
from .fill_playerscore_table import fill_playerscores_from_api
from ..framework.transaction_utils import update_team
from ..framework.utils import *
from ..framework.schema import session_scope


def main():

    parser = argparse.ArgumentParser(
        description="fill db tables with recent scores and transactions"
    )
    parser.add_argument(
        "--season", help="season, in format e.g. '1819'", default=CURRENT_SEASON
    )
    parser.add_argument(
        "--tag", help="identifying tag", default="AIrsenal" + CURRENT_SEASON
    )
    parser.add_argument(
        "--noattr", help="don't update player attributes", action="store_true"
    )

    args = parser.parse_args()

    season = args.season
    tag = args.tag
    do_attributes = not args.noattr

    with session_scope() as session:

        last_in_db = get_last_gameweek_in_db(season, session)
        last_finished = get_last_finished_gameweek()

        # TODO update players table

        if do_attributes:
            print("Updating attributes")
            fill_attributes_table_from_api(
                season, session, gw_start=last_in_db, gw_end=NEXT_GAMEWEEK
            )

        if last_finished > last_in_db:
            ## need to update
            fill_results_from_api(last_in_db + 1, NEXT_GAMEWEEK, season, session)
            fill_playerscores_from_api(season, session, last_in_db + 1, NEXT_GAMEWEEK)
        else:
            print("Matches and player-scores already up-to-date")
        ## now check transfers
        print("Checking team")
        db_players = sorted(get_current_players(season=season, dbsession=session))
        api_players = sorted(get_players_for_gameweek(last_finished))
        if db_players != api_players:
            update_team(season=season, session=session, verbose=True)
        else:
            print("Team is up-to-date")


if __name__ == "__main__":
    print(" ==== updating results and transactions === ")
    main()
