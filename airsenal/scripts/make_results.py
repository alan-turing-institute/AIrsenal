"""
Generate results CSV files from saved JSON files from fetcher.get_fixture_data()
"""
import json

import pandas as pd

SEASON = "2122"
FIXTURE_DATA_FILE = "../data/fixture_data_{}.json"
SUMMARY_DATA_FILE = "../data/FPL_{}.json"
RESULTS_FILE = "../data/results_{}.csv"
RESULTS_WITH_GW_FILE = "../data/results_{}_with_gw.csv"

keys_to_extract = {
    "kickoff_time": "date",
    "team_h": "home_team",
    "team_a": "away_team",
    "team_h_score": "home_score",
    "team_a_score": "away_score",
    "event": "gameweek",
}

with open(FIXTURE_DATA_FILE.format(SEASON), "r") as f:
    fixture_data = json.load(f)
with open(SUMMARY_DATA_FILE.format(SEASON), "r") as f:
    summary_data = json.load(f)

teams = {team["id"]: team["name"] for team in summary_data["teams"]}

fixtures_df = pd.DataFrame(fixture_data)
fixtures_df.rename(columns=keys_to_extract, inplace=True)
fixtures_df = fixtures_df[keys_to_extract.values()]

fixtures_df["home_team"].replace(teams, inplace=True)
fixtures_df["away_team"].replace(teams, inplace=True)

fixtures_df.to_csv(RESULTS_WITH_GW_FILE.format(SEASON), index=False)
fixtures_df.drop("gameweek", axis=1).to_csv(RESULTS_FILE.format(SEASON), index=False)

print(f"DONE {SEASON} SEASON!")
