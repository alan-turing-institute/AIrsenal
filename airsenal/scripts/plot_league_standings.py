#!/usr/bin/env python
"""
Plot the league
"""

import argparse

import matplotlib.pyplot as plt

from airsenal.framework.data_fetcher import FPLDataFetcher


def get_team_ids(league_data):
    return [team["entry"] for team in league_data["standings"]["results"]]


def get_team_names(league_data):
    return [team["entry_name"] for team in league_data["standings"]["results"]]


def get_team_history(team_data):
    output_dict = {"history": {}}
    for gw in team_data["current"]:
        output_dict["history"][gw["event"]] = {
            "points": gw["points"],
            "total_points": gw["total_points"],
            "ranking": gw["rank"],
            "overall_ranking": gw["overall_rank"],
        }

    return output_dict


def main():
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
