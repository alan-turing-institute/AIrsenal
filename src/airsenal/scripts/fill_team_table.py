"""
Fill the "Team" table with list of teams for all seasons, and the team_id which will
help fill other tables from raw json files
"""

import os

from sqlalchemy.orm.session import Session

from airsenal.framework.output import track
from airsenal.framework.schema import Team, get_session, session_scope
from airsenal.framework.season import CURRENT_SEASON, sort_seasons
from airsenal.framework.utils import get_past_seasons


def fill_team_table_from_file(filename: str, dbsession: Session | None = None) -> None:
    """
    use csv file
    """
    dbsession = dbsession if dbsession is not None else get_session()
    with open(filename) as infile:
        first_line = True
        for line in infile.readlines():
            if first_line:
                first_line = False
                continue
            t = Team()
            t.name, t.full_name, t.season, team_id = line.strip().split(",")
            t.team_id = int(team_id)
            dbsession.add(t)
    dbsession.commit()


def make_team_table(
    seasons: list[str] | None = None, dbsession: Session | None = None
) -> None:
    """
    Fill the db table containing the list of teams in the
    league for each season.
    """
    dbsession = dbsession if dbsession is not None else get_session()
    if seasons is None:
        seasons = []
    if not seasons:
        seasons = [CURRENT_SEASON]
        seasons += get_past_seasons(3)
    for season in track(sort_seasons(seasons), description="TEAMS"):
        filename = os.path.join(
            os.path.join(os.path.dirname(__file__), "..", "data", f"teams_{season}.csv")
        )
        fill_team_table_from_file(filename, dbsession=dbsession)


if __name__ == "__main__":
    with session_scope() as session:
        make_team_table(dbsession=session)
