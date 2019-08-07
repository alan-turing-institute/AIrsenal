#!/usr/bin/env python

"""
Fill the "Player" table with info from this and past seasonss FPL
"""
import os
import sys

import json

from ..framework.mappings import alternative_team_names, positions
from ..framework.schema import Player, PlayerAttributes, Base, engine
from ..framework.data_fetcher import FPLDataFetcher
from ..framework.utils import CURRENT_SEASON, get_past_seasons

def find_player_in_table(name, session):
    """
    see if we already have the player
    """
    player = session.query(Player).filter_by(name=name).first()
    return player if player else None


def num_players_in_table(session):
    """
    how many players already in player table
    """
    players = session.query(Player).all()
    return len(players)


def max_id_in_table(session):
    """
    Return the maximum ID in the player table
    """

    return session.query(Player).order_by(desc('player_id')).first().player_id


def fill_player_table_from_file(filename, season, session):
    """
    use json file
    """
    jplayers = json.load(open(filename))
    for i, jp in enumerate(jplayers):
        new_entry = False
        name = jp['name']
        print("{} adding {}".format(season, name))
        p = find_player_in_table(name, session)
        if not p:
            new_entry = True
            p = Player()
            p.player_id = max_id_in_table(session) + 1 # next id sequentially
            p.name = name
        pa = PlayerAttributes()
        pa.team = jp['team']
        pa.position = jp['position']
        pa.current_price = float(jp['cost'][1:])*10
        pa.season = season
        pa.gw_valid_from = 1 ### could potentially be superseded!
        p.attributes
        p.attributes.append(pa)
        session.add(pa)
        if new_entry:
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

        print("{} adding {}".format(season, name))
        p.name = name
        pa = PlayerAttributes()
        team_number = v["team"]
        for tk, tv in alternative_team_names.items():
            if str(team_number) in tv:
                pa.team = tk
                break
        pa.position = positions[v["element_type"]]
        pa.current_price = v["now_cost"]
        pa.season = season
        pa.gw_valid_from = 1 ### could potentially be superseded!
        p.attributes.append(pa)
        session.add(pa)
        session.add(p)
    session.commit()


def make_player_table(session):

    fill_player_table_from_api(CURRENT_SEASON,session)
    for season in get_past_seasons(3):
        filename = os.path.join( os.path.join(os.path.dirname(__file__),
                                              "..",
                                              "data",
                                              "player_summary_{}.json"\
                                              .format(season)))
        fill_player_table_from_file(filename,season,session)




if __name__ == "__main__":
    with session_scope() as session:
        make_player_table(session)
