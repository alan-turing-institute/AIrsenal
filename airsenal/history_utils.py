"""
functions to fill the db with player, match and fixture data
either from APIs (for current season) or from files (for past seasons).
"""

import os
import sys

import json

from .mappings import alternative_team_names, positions
from .schema import Player, Base, engine
from .data_fetcher import FPLDataFetcher

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DBSession = sessionmaker(bind=engine)
session = DBSession()


def fill_player_table_from_file(filename, season):
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


def fill_player_table_from_api(season):
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

    
