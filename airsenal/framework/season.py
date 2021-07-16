"""

Season details

"""
from datetime import datetime

from airsenal.framework.schema import Team, session


def get_current_season():
    """
    use the current time to find what season we're in.
    """
    current_time = datetime.now()
    if current_time.month > 6:
        start_year = current_time.year
    else:
        start_year = current_time.year - 1
    end_year = start_year + 1
    return "{}{}".format(str(start_year)[2:], str(end_year)[2:])


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
