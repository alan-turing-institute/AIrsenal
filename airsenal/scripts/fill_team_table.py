#!/usr/bin/env python

"""
Fill the "Team" table with list of teams for all seasons, and the team_id which will
help fill other tables from raw json files
"""
import os

from airsenal.framework.schema import Team
from airsenal.framework.utils import CURRENT_SEASON, get_past_seasons
from airsenal.framework.schema import session_scope, session


def fill_team_table_from_file(filename, dbsession=session):
    """
    use csv file
    """
    print("Filling Teams table from data in {}".format(filename))
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


def make_team_table(seasons=[], dbsession=session):
    """
    Fill the db table containing the list of teams in the
    league for each season.
    """

    if not seasons:
        seasons = [CURRENT_SEASON]
        seasons += get_past_seasons(4)
    for season in seasons:
        filename = os.path.join(
            os.path.join(
                os.path.dirname(__file__), "..", "data", "teams_{}.csv".format(season)
            )
        )
        fill_team_table_from_file(filename, dbsession=dbsession)


if __name__ == "__main__":
    with session_scope() as session:
        make_team_table(dbsession=session)
