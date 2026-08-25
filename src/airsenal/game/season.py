"""
Which season it is, and how a season is written.

A season is the string `"2122"` for 2021/22 - the last two digits of each year.
"""

from datetime import datetime


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
    """Convert a season in "1819" format to the year it started (2018)."""
    return int(f"20{season[:2]}")


def sort_seasons(seasons: list[str], desc: bool = True) -> list[str]:
    """
    Sort season strings in "1819" format chronologically.

    Args:
        desc: If True, the default, most recent season first.
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
