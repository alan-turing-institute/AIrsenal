"""
Which season it is, and how a season is written.

A season is the string `"2122"` for 2021/22 - the last two digits of each year.
"""

from datetime import datetime


def get_current_season() -> str:
    """The season we are currently in, from the current date."""
    current_time = datetime.now()
    start_year = current_time.year if current_time.month > 5 else current_time.year - 1
    end_year = start_year + 1
    return f"{str(start_year)[2:]}{str(end_year)[2:]}"


CURRENT_SEASON = get_current_season()


# The first season the FPL API reported expected goals and assists for.
FIRST_SEASON_WITH_EXPECTED_GOALS = "2223"


def season_str_to_year(season: str) -> int:
    """Convert a season in "1819" format to the year it started (2018)."""
    return int(f"20{season[:2]}")


def sort_seasons(seasons: list[str], desc: bool = True) -> list[str]:
    """Sort season strings chronologically, most recent first unless `desc` is False."""
    return sorted(seasons, key=season_str_to_year, reverse=desc)


def get_next_season(season: str) -> str:
    """The season after this one: '1819' becomes '1920'."""
    return f"{int(season[:2]) + 1:02d}{int(season[2:]) + 1:02d}"


def get_previous_season(season: str) -> str:
    """The season before this one: '1819' becomes '1718'."""
    start_year = int(season[:2])
    end_year = int(season[2:])
    prev_start_year = start_year - 1
    prev_end_year = end_year - 1
    return f"{prev_start_year}{prev_end_year}"


def get_past_seasons(num_seasons: int) -> list[str]:
    """The `num_seasons` seasons before the current one, most recent first."""
    season = CURRENT_SEASON
    seasons = []
    for _ in range(num_seasons):
        season = get_previous_season(season)
        seasons.append(season)
    return seasons


def default_seasons() -> list[str]:
    """The current season and the three before it, most recent first."""
    return [CURRENT_SEASON, *get_past_seasons(3)]
