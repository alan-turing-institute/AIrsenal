#!/usr/bin/env python

"""
Fill the "Team" table with list of teams for all seasons, and the team_id which will
help fill other tables from raw json files
"""
import os
from typing import List, Optional

from sqlalchemy.orm.session import Session

from airsenal.framework.schema import Team, session, session_scope
from airsenal.framework.season import CURRENT_SEASON, sort_seasons
from airsenal.framework.utils import get_past_seasons


def fill_team_table_from_file(filename: str, dbsession: Session = session) -> None:
    """
    use csv file
    """
    print(f"Filling Teams table from data in {filename}")
    infile = open(filename)
    first_line = True
    for line in infile.readlines():
        if first_line:
            first_line = False
            continue
        t = Team()
        t.name, t.full_name, t.season, t.team_id = line.strip().split(",")
        print(t.name, t.full_name, t.season, t.team_id)
        dbsession.add(t)
    dbsession.commit()


def make_team_table(
    seasons: Optional[List[str]] = [], dbsession: Session = session
) -> None:
    """
    Fill the db table containing the list of teams in the
    league for each season.
    """
    if not seasons:
        seasons = [CURRENT_SEASON]
        seasons += get_past_seasons(3)
    for season in sort_seasons(seasons):
        filename = os.path.join(
            os.path.join(os.path.dirname(__file__), "..", "data", f"teams_{season}.csv")
        )
        fill_team_table_from_file(filename, dbsession=dbsession)


if __name__ == "__main__":
    with session_scope() as session:
        make_team_table(dbsession=session)
