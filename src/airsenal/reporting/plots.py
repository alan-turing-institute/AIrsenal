"""Plot a mini-league's metrics by gameweek."""

from typing import Any

import matplotlib.pyplot as plt

from airsenal.remote.fpl_api import get_fetcher


def get_team_ids(league_data: dict[str, Any]) -> list[int]:
    return [team["entry"] for team in league_data["standings"]["results"]]


def get_team_names(league_data: dict[str, Any]) -> list[str]:
    return [team["entry_name"] for team in league_data["standings"]["results"]]


def get_team_history(team_data: dict[str, Any]) -> dict[str, Any]:
    # not dict[str, dict[int, dict[str, int]]]: the caller adds a "name" string
    # to the same dict alongside "history".
    output_dict: dict[str, Any] = {"history": {}}
    for gw in team_data["current"]:
        output_dict["history"][gw["event"]] = {
            "points": gw["points"],
            "total_points": gw["total_points"],
            "ranking": gw["rank"],
            "overall_ranking": gw["overall_rank"],
        }

    return output_dict


def plot_standings(thing_to_plot: str) -> None:
    """Plot a selected mini-league metric by gameweek."""
    fetcher = get_fetcher()
    league_data = fetcher.get_fpl_league_data()
    if league_data is None:
        msg = "Could not retrieve league data from the FPL API"
        raise RuntimeError(msg)
    team_ids = get_team_ids(league_data)
    team_names = get_team_names(league_data)
    team_histories = []
    for i, team_id in enumerate(team_ids):
        team_data = fetcher.get_fpl_team_history_data(team_id)
        history_dict = get_team_history(team_data)
        history_dict["name"] = team_names[i]
        team_histories.append(history_dict)

    xvals = sorted(team_histories[0]["history"].keys())
    points = []
    for th in team_histories:
        points.append(
            [th["history"][gw][thing_to_plot] for gw in sorted(th["history"].keys())]
        )
        plt.plot(xvals, points[-1], label=th["name"])
    plt.legend(loc="best")
    plt.xlabel("gameweek")
    plt.ylabel(thing_to_plot)
    plt.show()
