#!/usr/bin/env python

"""
Fill the "Player" table with info from this seasons FPL
(FPL_2017-18.json).
"""

from ..framework.mappings import alternative_team_names, positions
from ..framework.schema import Player, session_scope
from ..framework.data_fetcher import FPLDataFetcher


def make_player_table(session):
    # fill up the player table
    data_fetcher = FPLDataFetcher()
    player_dict = data_fetcher.get_player_summary_data()

    for k, v in player_dict.items():
        p = Player()
        p.player_id = k
        name = "{} {}".format(v["first_name"], v["second_name"])
        print("Adding {}".format(name))
        p.name = name
        team_number = v["team"]
        for tk, tv in alternative_team_names.items():
            if str(team_number) in tv:
                p.team = tk
                break
        p.position = positions[v["element_type"]]
        p.current_price = v["now_cost"]
        session.add(p)
    session.commit()


if __name__ == "__main__":
    with session_scope() as session:
        make_player_table(session)
