"""
Fill the "Team" table with list of teams for all seasons, and the team_id which will
help fill other tables from raw json files
"""

import os

from sqlalchemy import select
from sqlalchemy.orm.session import Session

from airsenal.framework.data_fetcher import FPLDataFetcher
from airsenal.framework.schema import Team, session, session_scope
from airsenal.framework.season import CURRENT_SEASON, sort_seasons
from airsenal.framework.utils import get_past_seasons


def fill_team_table_from_file(filename: str, dbsession: Session = session) -> None:
    """
    use csv file
    """
    print(f"Filling Teams table from data in {filename}")
    with open(filename) as infile:
        first_line = True
        for line in infile.readlines():
            if first_line:
                first_line = False
                continue
            t = Team()
            t.name, t.full_name, t.season, team_id = line.strip().split(",")
            print(t.name, t.full_name, t.season, team_id)
            t.team_id = int(team_id)
            dbsession.add(t)
    dbsession.commit()


def fill_team_table_from_api(
    season: str = CURRENT_SEASON,
    dbsession: Session = session,
    apifetcher: FPLDataFetcher | None = None,
) -> int:
    """
    Fill the teams for a season from the FPL API.

    The data directory only carries a CSV for seasons someone has already written
    one for, so a new season would otherwise have no teams at all - and every
    fixture lookup goes through team_id, so nothing else can be filled either.
    """
    if apifetcher is None:
        apifetcher = FPLDataFetcher()
    teams = apifetcher.get_current_summary_data()["teams"]
    if not teams:
        msg = f"No teams returned by the API for {season}"
        raise RuntimeError(msg)

    existing = {
        t.team_id: t
        for t in dbsession.scalars(select(Team).where(Team.season == season)).all()
    }
    added = 0
    for team in teams:
        t = existing.get(team["id"])
        if t is None:
            t = Team()
            t.team_id = team["id"]
            t.season = season
            dbsession.add(t)
            added += 1
        t.name = team["short_name"]
        t.full_name = team["name"]
    dbsession.commit()
    print(f"Filled {added} teams for {season} from the API")
    return added


def make_team_table(
    seasons: list[str] | None = None, dbsession: Session = session
) -> None:
    """
    Fill the db table containing the list of teams in the
    league for each season.
    """
    if seasons is None:
        seasons = []
    if not seasons:
        seasons = [CURRENT_SEASON]
        seasons += get_past_seasons(3)
    for season in sort_seasons(seasons):
        filename = os.path.join(
            os.path.join(os.path.dirname(__file__), "..", "data", f"teams_{season}.csv")
        )
        if os.path.exists(filename):
            fill_team_table_from_file(filename, dbsession=dbsession)
        elif season == CURRENT_SEASON:
            fill_team_table_from_api(season, dbsession=dbsession)
        else:
            msg = (
                f"No teams CSV for {season}, and only the current season is "
                "available from the API"
            )
            raise FileNotFoundError(msg)


if __name__ == "__main__":
    with session_scope() as session:
        make_team_table(dbsession=session)
