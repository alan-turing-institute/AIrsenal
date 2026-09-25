"""Plot a mini-league's metrics by gameweek."""

from typing import Any

import matplotlib.pyplot as plt

from airsenal.remote.fpl_api import get_fetcher


def get_team_history(team_data: dict[str, Any]) -> dict[str, Any]:
    output_dict: dict[str, Any] = {"history": {}}
    for entry in team_data["current"]:
        output_dict["history"][entry["event"]] = {
            "points": entry["points"],
            "total_points": entry["total_points"],
            "ranking": entry["rank"],
            "overall_ranking": entry["overall_rank"],
        }

    return output_dict


def plot_standings(thing_to_plot: str) -> None:
    """Plot a selected mini-league metric by gameweek."""
    fetcher = get_fetcher()
    league_data = fetcher.get_fpl_league_data()
    if league_data is None:
        msg = "Could not retrieve league data from the FPL API"
        raise RuntimeError(msg)
    team_histories = []
    for team in league_data["standings"]["results"]:
        history_dict = get_team_history(
            fetcher.get_fpl_team_history_data(team["entry"])
        )
        history_dict["name"] = team["entry_name"]
        team_histories.append(history_dict)

    xvals = sorted(team_histories[0]["history"].keys())
    for th in team_histories:
        values = [
            th["history"][gameweek][thing_to_plot]
            for gameweek in sorted(th["history"].keys())
        ]
        plt.plot(xvals, values, label=th["name"])
    plt.legend(loc="best")
    plt.xlabel("gameweek")
    plt.ylabel(thing_to_plot)
    plt.show()
