"""

Season details

"""
from datetime import datetime
from airsenal.framework.data_fetcher import FPLDataFetcher


def get_current_season():
    """
    use the current time to find what season we're in.
    """
    current_time = datetime.now()
    if current_time.month > 7:
        start_year = current_time.year
    else:
        start_year = current_time.year - 1
    end_year = start_year + 1
    return "{}{}".format(str(start_year)[2:], str(end_year)[2:])


# make this a global variable in this module, import into other modules
CURRENT_SEASON = get_current_season()


fetcher = FPLDataFetcher()
# TODO make this a database table so we can look at past seasons
CURRENT_TEAMS = [t["short_name"] for t in fetcher.get_current_team_data().values()]
