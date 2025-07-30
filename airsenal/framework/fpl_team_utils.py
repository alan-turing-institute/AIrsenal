"""
Functions to get data on specified FPL teams and leagues
"""

from airsenal.framework.utils import fetcher


def get_overall_points(gameweek=None):
    """
    Get our total points
    """
    data = fetcher.get_fpl_team_data(gameweek=gameweek)
    if not gameweek:
        return data["entry"]["summary_overall_points"]
    if isinstance(gameweek, int) and gameweek <= len(data["history"]):
        return data["history"][gameweek - 1]["points"]
    print("Unknown gameweek")
    return 0


def get_overall_ranking(gameweek=None):
    """
    Get our overall ranking
    """
    data = fetcher.get_fpl_team_data(gameweek=gameweek)
    if not gameweek:
        return data["entry"]["summary_overall_rank"]
    if isinstance(gameweek, int) and gameweek <= len(data["history"]):
        return data["history"][gameweek - 1]["rank"]
    print("Unknown gameweek")
    return 0


def get_league_standings():
    """
    Get stuff about our mini-league
    """
    data = fetcher.get_fpl_league_data()
    team_name = data["league"]["name"]
    standings = [
        {"name": s["entry_name"], "manager": s["player_name"], "points": s["total"]}
        for s in data["standings"]["results"]
    ]

    return team_name, standings
