#!/usr/bin/env python

"""
Fill the "fifa_ratings" table with info from FIFA 19
(fifa_team_ratings.csv).
"""

import sys

from ..framework.mappings import alternative_team_names
from ..framework.schema import FifaTeamRating, Base, engine

from sqlalchemy.orm import sessionmaker

DBSession = sessionmaker(bind=engine)
session = DBSession()

if __name__ == "__main__":

    input_file = open("../data/fifa_team_ratings.csv")
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
