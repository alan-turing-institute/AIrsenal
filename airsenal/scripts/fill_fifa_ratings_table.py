#!/usr/bin/env python

"""
Fill the "fifa_ratings" table with info from FIFA 19
(fifa_team_ratings.csv).
"""

import os

from ..framework.mappings import alternative_team_names
from ..framework.schema import FifaTeamRating, session_scope
from ..framework.utils import CURRENT_SEASON


def make_fifa_ratings_table(session, season=CURRENT_SEASON):
    # make the fifa ratings table
    # TODO: scrape the data first rather than committing file to repo
    input_path = os.path.join(
        os.path.dirname(__file__), "../data/fifa_team_ratings_{}.csv".format(season)
    )
    input_file = open(input_path)
    for line in input_file.readlines()[1:]:
        team, att, mid, defn, ovr = line.strip().split(",")
        print(line.strip())
        r = FifaTeamRating()
        r.team = team
        r.att = int(att)
        r.defn = int(defn)
        r.mid = int(mid)
        r.ovr = int(ovr)
        team_is_known = False
        for k, v in alternative_team_names.items():
            if team in v:
                r.team = k
                team_is_known = True
            elif team == k:
                team_is_known = True
        if not team_is_known:
            raise ValueError("Unknown team {}.".format(team))
        session.add(r)
    session.commit()


if __name__ == "__main__":
    with session_scope() as session:
        make_fifa_ratings_table(session)
