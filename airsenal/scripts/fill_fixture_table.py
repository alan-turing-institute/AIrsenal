#!/usr/bin/env python

"""
Fill the "fixture" table with info from this seasons FPL
(fixtures.csv).
"""

import os

from ..framework.mappings import alternative_team_names
from ..framework.schema import Fixture, session_scope


def get_fixture_list():
    # TODO: get this from the footballdata API
    input_path = os.path.join(os.path.dirname(__file__), "../data/fixtures.csv")
    return open(input_path)


def make_fixture_table(session):
    # fill the fixture table
    input_file = get_fixture_list()
    for line in input_file.readlines()[1:]:
        gameweek, date, home_team, away_team = line.strip().split(",")
        print(line.strip())
        f = Fixture()
        f.gameweek = int(gameweek)
        f.date = date
        for k, v in alternative_team_names.items():
            if home_team in v:
                f.home_team = k
            elif away_team in v:
                f.away_team = k
        session.add(f)
    session.commit()


if __name__ == "__main__":
    with session_scope() as session:
        make_fixture_table(session)

