"""
Fill the "team" table with the teams in each season.

Also the per-season team_id the raw JSON files key on, which is what lets the
other tables be filled from them.
"""

from sqlalchemy.orm.session import Session

from airsenal.core.console import track
from airsenal.core.data_files import FilePath, data_file
from airsenal.db.models import Team
from airsenal.db.session import get_session
from airsenal.game.season import CURRENT_SEASON, get_past_seasons, sort_seasons


def fill_team_table_from_file(
    filename: FilePath, dbsession: Session | None = None
) -> None:
    """Read the teams for a season from its packaged CSV file."""
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
    """Fill the team table with the league's teams for every season."""
    dbsession = dbsession if dbsession is not None else get_session()
    if seasons is None:
        seasons = []
    if not seasons:
        seasons = [CURRENT_SEASON]
        seasons += get_past_seasons(3)
    for season in track(sort_seasons(seasons), description="TEAMS"):
        fill_team_table_from_file(data_file(f"teams_{season}.csv"), dbsession=dbsession)
