#!/usr/bin/env python

"""
Fill the "Team" table with list of teams for all seasons, and the team_id which will help
fill other tables from raw json files
"""
import os
import sys

import json
from sqlalchemy import desc

from ..framework.schema import Team
from ..framework.data_fetcher import FPLDataFetcher
from ..framework.utils import CURRENT_SEASON, get_past_seasons
from ..framework.schema import session_scope


def fill_team_table_from_file(filename, session):
    """
    use csv file
    """
    print("Filling Teams table from data in {}".format(filename))
    infile = open(filename)
    first_line = True
    for line in infile.readlines():
        if first_line:
            first_line = False
            continue
        t = Team()
        t.name, t.full_name, t.season, t.team_id = line.strip().split(",")
        print(t.name, t.full_name, t.season, t.team_id)
        session.add(t)
    session.commit()


def make_team_table(session):

    seasons = [CURRENT_SEASON]
    seasons += get_past_seasons(4)
    for season in seasons:
        filename = os.path.join( os.path.join(os.path.dirname(__file__),
                                              "..",
                                              "data",
                                              "teams_{}.csv"\
                                              .format(season)))
        fill_team_table_from_file(filename, session)




if __name__ == "__main__":
    with session_scope() as session:
        make_team_table(session)
