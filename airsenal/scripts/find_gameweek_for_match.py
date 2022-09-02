#!/usr/bin/env python

import json
import sys

import pandas as pd

"""
The results_xxyy.csv don't have gameweek info in, create them by looking
at gameweek dates in FPL_xxyy.json file.

!!! WARNING !!! FPL_xxyy.json file must be the latest available version as
deadlines can change during the season. Using an out of date version can
therefore lead to matches being assigned to wrong gameweek.
It may also be more robust to fix previous version of this file using player
score files, rather than relying on the approach used here.
"""


def get_gameweek_deadlines(fpl_file_path):
    with open(fpl_file_path, "r") as f:
        fpl_json = json.load(f)

    deadlines = pd.Series({e["id"]: e["deadline_time"] for e in fpl_json["events"]})

    deadlines = pd.to_datetime(deadlines).dt.date
    deadlines.sort_index(inplace=True)

    return deadlines


def get_gameweek_from_date(date, deadlines):
    date = pd.to_datetime(date, dayfirst=True).date()

    gw = deadlines[date >= deadlines]
    print(f"GW{gw.index[-1]} (deadline {gw.values[-1]})")
    return gw.index[-1]


if __name__ == "__main__":
    season = sys.argv[-1]

    results_file = open(f"../data/results_{season}.csv")
    with open(f"../data/results_{season}_with_gw.csv", "w") as output_file:
        fpl_file_path = f"../data/FPL_{season}.json"

        deadlines = get_gameweek_deadlines(fpl_file_path)

        for linecount, line in enumerate(results_file.readlines()):
            if linecount == 0:
                output_file.write(line.strip() + ",gameweek\n")
                continue

            date, home_team, away_team = line.split(",")[:3]
            print(date, home_team, away_team)
            gameweek = get_gameweek_from_date(date, deadlines)
            output_file.write(line.strip() + "," + str(gameweek) + "\n")

        results_file.close()
