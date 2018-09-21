#!/usr/bin/env python
"""
Plot the league
"""

import sys

sys.path.append("..")

import argparse
import matplotlib.pyplot as plt

from framework.data_fetcher import FPLDataFetcher


def get_team_ids(league_data):
    team_ids = []
    for team in league_data["standings"]["results"]:
        team_ids.append(team["entry"])
    return team_ids


def get_team_history(team_data):
    output_dict = {}
    output_dict["name"] = team_data["entry"]["name"]
    output_dict["history"] = {}
    for gw in team_data["history"]:
        output_dict["history"][gw["event"]] = {}
        output_dict["history"][gw["event"]]["points"] = gw["points"]
        output_dict["history"][gw["event"]]["total_points"] = gw["total_points"]
        output_dict["history"][gw["event"]]["ranking"] = gw["rank"]
        output_dict["history"][gw["event"]]["overall_ranking"] = gw["overall_rank"]
    return output_dict


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="plot mini-league")
    parser.add_argument(
        "--thing_to_plot",
        help="points,total_points,ranking,overall_ranking",
        default="total_points",
    )
    args = parser.parse_args()
    thing_to_plot = args.thing_to_plot

    fetcher = FPLDataFetcher()
    league_data = fetcher.get_fpl_league_data()
    team_ids = get_team_ids(league_data)
    team_histories = []
    for team_id in team_ids:
        team_data = fetcher.get_fpl_team_data(team_id)
        team_histories.append(get_team_history(team_data))

    xvals = sorted(team_histories[0]["history"].keys())
    points = []
    for th in team_histories:
        points.append(
            [th["history"][gw][thing_to_plot] for gw in sorted(th["history"].keys())]
        )
        plt.plot(xvals, points[-1], label=th["name"])
    plt.legend(loc="best")
    plt.show()
