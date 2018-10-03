#!/usr/bin/env python

"""
Fill the "Player" table with info from this seasons FPL
(FPL_2017-18.json).
"""
import os

from ..framework.history_utils import (
    fill_player_table_from_api,
    fill_player_table_from_file
)


def make_player_table(session):

    fill_player_table_from_api(session,"1819")
    for season in ["1718","1617","1516"]:
        filename = os.path.join("airsenal/data","player_summary_{}.json".format(season))
        fill_player_table_from_file(session,filename,season)

if __name__ == "__main__":
    with session_scope() as session:
        make_player_table(session)
