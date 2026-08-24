"""

Season details

"""

from datetime import datetime

import pandas as pd


def get_current_season() -> str:
    """
    use the current time to find what season we're in.
    """
    current_time = datetime.now()
    start_year = current_time.year if current_time.month > 5 else current_time.year - 1
    end_year = start_year + 1
    return f"{str(start_year)[2:]}{str(end_year)[2:]}"


# make this a global variable in this module, import into other modules
CURRENT_SEASON = get_current_season()


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


def sort_seasons(seasons: list[str], desc: bool = True) -> list[str]:
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


def get_next_season(season: str) -> str:
    """
    Convert string e.g. '1819' into one for next season, i.e. '1920'.
    """
    start_year = int(season[:2])
    end_year = int(season[2:])
    next_start_year = (
        f"0{start_year + 1}" if start_year + 1 < 10 else str(start_year + 1)
    )
    next_end_year = f"0{end_year + 1}" if end_year + 1 < 10 else str(end_year + 1)
    return f"{next_start_year}{next_end_year}"


def get_start_end_dates_of_season(season: str) -> list[pd.Timestamp]:
    """
    Obtains rough start and end dates for the season.
    Takes into account the shorter and longer seasons in 19/20 and 20/21.
    """
    start_year = int(f"20{season[:2]}")
    end_year = int(f"20{season[2:]}")
    if season == "1920":
        # regular start, late end to season
        return [pd.Timestamp(2019, 7, 1), pd.Timestamp(2020, 7, 31)]
    if season == "2021":
        # late start to season, regular end
        return [pd.Timestamp(2020, 8, 1), pd.Timestamp(2021, 6, 30)]
    # regular season
    return [pd.Timestamp(start_year, 7, 1), pd.Timestamp(end_year, 6, 30)]


def get_previous_season(season: str) -> str:
    """
    Convert string e.g. '1819' into one for previous season, i.e. '1718'
    """
    start_year = int(season[:2])
    end_year = int(season[2:])
    prev_start_year = start_year - 1
    prev_end_year = end_year - 1
    return f"{prev_start_year}{prev_end_year}"


def get_past_seasons(num_seasons: int) -> list[str]:
    """
    Go back num_seasons from the current one.
    """
    season = CURRENT_SEASON
    seasons = []
    for _ in range(num_seasons):
        season = get_previous_season(season)
        seasons.append(season)
    return seasons
