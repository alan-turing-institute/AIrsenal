#!/usr/bin/env python

"""
Fill the "fifa_ratings" table with info from FIFA 19
(fifa_team_ratings.csv).
"""

import os

from airsenal.framework.mappings import alternative_team_names
from airsenal.framework.schema import FifaTeamRating, session_scope
from airsenal.framework.utils import CURRENT_SEASON, get_past_seasons


def make_fifa_ratings_table(session):
    # make the fifa ratings table
    # TODO: scrape the data first rather than committing file to repo

    seasons = get_past_seasons(3)
    seasons.append(CURRENT_SEASON)

    for season in seasons:
        print("FIFA RATINGS {}".format(season))
        input_path = os.path.join(
            os.path.dirname(__file__), "../data/fifa_team_ratings_{}.csv".format(season)
        )
        try:
            input_file = open(input_path)
        except FileNotFoundError:
            print("!!! No FIFA ratings file found for {}".format(season))
            continue

        for line in input_file.readlines()[1:]:
            team, att, mid, defn, ovr = line.strip().split(",")
            r = FifaTeamRating()
            r.season = season
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
