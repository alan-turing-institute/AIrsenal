#!/usr/bin/env python

"""
Fill the "Player" table with info from this and past seasonss FPL
"""
import os
import json

from airsenal.framework.schema import Player, session_scope
from airsenal.framework.data_fetcher import FPLDataFetcher
from airsenal.framework.utils import CURRENT_SEASON, get_past_seasons


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


def fill_player_table_from_file(filename, season, session):
    """
    use json file
    """
    jplayers = json.load(open(filename))
    n_new_players = 0
    for i, jp in enumerate(jplayers):
        new_entry = False
        name = jp["name"]
        print("PLAYER {} {}".format(season, name))
        p = find_player_in_table(name, session)
        if not p:
            n_new_players += 1
            new_entry = True
            p = Player()
#            p.player_id = (
#                max_id_in_table(session) + n_new_players
#            )  # next id sequentially
            p.name = name
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
        p.fpl_api_id = k
        first_name = v["first_name"]  # .encode("utf-8")
        second_name = v["second_name"]  # .encode("utf-8")
        name = "{} {}".format(first_name, second_name)

        print("PLAYER {} {}".format(season, name))
        p.name = name
        session.add(p)
    session.commit()


def make_player_table(session):

    fill_player_table_from_api(CURRENT_SEASON, session)
    for season in get_past_seasons(3):
        filename = os.path.join(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "data",
                "player_summary_{}.json".format(season),
            )
        )
        fill_player_table_from_file(filename, season, session)


if __name__ == "__main__":
    with session_scope() as session:
        make_player_table(session)
