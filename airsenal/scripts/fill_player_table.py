#!/usr/bin/env python

"""
Fill the "Player" table with info from this seasons FPL
(FPL_2017-18.json).
"""
import os
import sys

import json

from ..framework.mappings import alternative_team_names, positions
from ..framework.schema import Player, Base, engine
from ..framework.data_fetcher import FPLDataFetcher


def fill_player_table_from_file(filename, season, session):
    """
    use json file
    """
    jplayers = json.load(open(filename))
    for i, jp in enumerate(jplayers):
        p = Player()
        p.player_id = i
        p.name = jp['name']
        p.team = jp['team']
        p.position = jp['position']
        p.current_price = float(jp['cost'][1:])*10
        p.season = season
        session.add(p)
    session.commit()


def fill_player_table_from_api(season, session):
    """
    use the FPL API
    """
    df = FPLDataFetcher()
    pd = df.get_player_summary_data()

    for k, v in pd.items():
        p = Player()
        p.player_id = k
        first_name = v["first_name"]#.encode("utf-8")
        second_name = v["second_name"]#.encode("utf-8")
        name = "{} {}".format(first_name,second_name)
       # print("Adding {}".format(name))
        p.name = name
        team_number = v["team"]
        for tk, tv in alternative_team_names.items():
            if str(team_number) in tv:
                p.team = tk
                break
        p.position = positions[v["element_type"]]
        p.current_price = v["now_cost"]
        p.season = season
        session.add(p)
    session.commit()


def make_player_table(session):

    fill_player_table_from_api("1819",session)
    for season in ["1718","1617","1516"]:
        filename = os.path.join( os.path.join(os.path.dirname(__file__),
                                              "..",
                                              "data",
                                              "player_summary_{}.json"\
                                              .format(season)))
        fill_player_table_from_file(filename,season,session)




if __name__ == "__main__":
    with session_scope() as session:
        make_player_table(session)
