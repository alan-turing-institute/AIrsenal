#!/usr/bin/env python

"""
Fill the "fifa_ratings" table with info from FIFA 19
(fifa_team_ratings.csv).
"""

import os
from typing import List, Optional

from sqlalchemy.orm.session import Session

from airsenal.framework.mappings import alternative_team_names
from airsenal.framework.schema import FifaTeamRating, session, session_scope
from airsenal.framework.season import CURRENT_SEASON, sort_seasons
from airsenal.framework.utils import get_past_seasons


def make_fifa_ratings_table(
    seasons: Optional[List[str]] = [], dbsession: Session = session
) -> None:
    # make the fifa ratings table
    # TODO: scrape the data first rather than committing file to repo

    if not seasons:
        seasons = [CURRENT_SEASON]
        seasons += get_past_seasons(3)
    for season in sort_seasons(seasons):
        print(f"FIFA RATINGS {season}")
        input_path = os.path.join(
            os.path.dirname(__file__), f"../data/fifa_team_ratings_{season}.csv"
        )
        try:
            input_file = open(input_path)
        except FileNotFoundError:
            print(f"!!! No FIFA ratings file found for {season}")
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
                raise ValueError(f"Unknown team {team}.")
            dbsession.add(r)
    dbsession.commit()


if __name__ == "__main__":
    with session_scope() as session:
        make_fifa_ratings_table(dbsession=session)
