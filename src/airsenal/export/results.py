"""
Generate results CSV files from saved JSON files from fetcher.get_fixture_data()
"""

import json

import pandas as pd

from airsenal.core.logging import get_logger
from airsenal.core.resources import resource
from airsenal.core.season import CURRENT_SEASON

logger = get_logger(__name__)

FIXTURE_DATA_FILE = str(resource("fixture_data_{}.json"))
SUMMARY_DATA_FILE = str(resource("FPL_{}.json"))
RESULTS_FILE = str(resource("results_{}.csv"))

keys_to_extract = {
    "kickoff_time": "date",
    "team_h": "home_team",
    "team_a": "away_team",
    "team_h_score": "home_score",
    "team_a_score": "away_score",
    "event": "gameweek",
}


def make_results(season):
    with open(FIXTURE_DATA_FILE.format(season)) as f:
        fixture_data = json.load(f)
    with open(SUMMARY_DATA_FILE.format(season)) as f:
        summary_data = json.load(f)

    teams = {team["id"]: team["name"] for team in summary_data["teams"]}

    fixtures_df = pd.DataFrame(fixture_data)
    fixtures_df.rename(columns=keys_to_extract, inplace=True)
    fixtures_df = fixtures_df[keys_to_extract.values()]

    fixtures_df["home_team"].replace(teams, inplace=True)
    fixtures_df["away_team"].replace(teams, inplace=True)

    fixtures_df.to_csv(RESULTS_FILE.format(season), index=False)
    logger.info("Made results file for %s season!", season)


if __name__ == "__main__":
    make_results(CURRENT_SEASON)
