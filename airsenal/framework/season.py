"""

Season details

"""
from datetime import datetime
from typing import List

from airsenal.framework.schema import Team, session


def get_current_season():
    """
    use the current time to find what season we're in.
    """
    current_time = datetime.now()
    if current_time.month > 5:
        start_year = current_time.year
    else:
        start_year = current_time.year - 1
    end_year = start_year + 1
    return f"{str(start_year)[2:]}{str(end_year)[2:]}"


# make this a global variable in this module, import into other modules
CURRENT_SEASON = get_current_season()


def get_teams_for_season(season, dbsession):
    """
    Query the Team table and get a list of teams for a given
    season.
    """
    teams = dbsession.query(Team).filter_by(season=season).all()
    return [t.name for t in teams]


# global variable for the module
CURRENT_TEAMS = get_teams_for_season(CURRENT_SEASON, session)


def season_str_to_year(season: str) -> int:
    """Convert season in "1819" format to the year the season started (2018)

    Parameters
    ----------
    season : str
        Season string in "1819" format (for 2018/19 season)

    Returns
    -------
    int
        Year season started
    """
    return int(f"20{season[:2]}")


def sort_seasons(seasons: List[str], desc: bool = True) -> List[str]:
    """_summary_

    Parameters
    ----------
    seasons : List[str]
        List of seasons strings in "1819" formrat (for 2018/19 season)

    desc : bool , optional
        If True, sort from most recent season to oldest. By default True.

    Returns
    -------
    List[str]
        Seasons sorted in chronological order (by default from most recent to oldest)
    """
    return sorted(seasons, key=season_str_to_year, reverse=desc)
