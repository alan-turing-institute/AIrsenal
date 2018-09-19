#!/usr/bin/env python

"""
simple script, check whether recent matches have been played since
the last entries in the DB.
"""

import sys
sys.path.append("..")

from fill_match_table import fill_table_from_list, fill_from_api
from fill_playerscore_this_season import fill_playerscore_table
from framework.utils import *

if __name__ == "__main__":
    last_in_db = get_last_gameweek_in_db()
    last_finished = get_last_finished_gameweek()
    if last_finished > last_in_db:
        ## need to update
        next_gw = get_next_gameweek()
        fill_from_api(last_in_db+1, next_gw)

        fill_playerscore_table(last_in_db+1, next_gw)
    else:
        print("Already up-to-date")
