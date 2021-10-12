#!/usr/bin/env python

"""
Use BeautifulSoup to follow links to scrape data from understat.com

To setup -
`pip install bs4`
"""

import json
import argparse
import requests

from tqdm import tqdm
from bs4 import BeautifulSoup

LEAGUE_URL = "https://understat.com/league/epl/{}"
MATCH_URL = "https://understat.com/match/{}"
base_url = {
    "1516": LEAGUE_URL.format("2015"),
    "1617": LEAGUE_URL.format("2016"),
    "1718": LEAGUE_URL.format("2017"),
    "1819": LEAGUE_URL.format("2018"),
    "1920": LEAGUE_URL.format("2019"),
    "2021": LEAGUE_URL.format("2020"),
}


def get_matches_info(season: str):
    """Get the basic information for each match of the `season` from understat.com

    Parameters
    ----------
    season : str
        String corresponding to the season for which we need to find the
        match info.

    Returns
    -------
    list
        List of dictionaries containing the following information for
        each match:
        `id`: ID of the match used by understat.com
        `isResult`: True if the object is the result of a match.
        `h`: Dictionary object containing the home team information.
        `a`: Dictionary object containing the away team information.
        `goals`: Dictionary with number ofgoals scored by the home and
                away teams.
        `xG`: The xG statistic for both teams.
        `datetime`: The date and time of the match.
        `forecast`: Forecasted values for win/loss/draw.
    """
    response = requests.get(base_url.get(season, "-1"))
    if response.ok:
        html = response.text
        start = html.find("JSON") + 11
        end = html.find(")", start)
        json_string = html[start:end]
        json_string = json_string.encode("utf-8").decode("unicode_escape")

        matches_list = json.loads(json_string[1:-1])
        return matches_list
    else:
        raise ValueError(
            f"Please provide valid season to scrape data. {season} not in {list(base_url.keys())}"
        )


def parse_match(match_info: dict):
    """Parse match webpage

    This function parses the webpage for the match corresponding to
    `match_id` and returns a dictionary with the required information.

    Parameters
    ----------
    match_id: dict
        A dictionary that contains the basic information
        regarding the match like `id`, `h` (home_team), `a` (away_team)


    Returns
    -------
    dict
        Dictionary with the following structure:
        {
            "home": home_team,
            "away": away_team,
            "goals": (list) goals,
            "subs": (list) subs,
        }

    """
    match_id = match_info.get("id", None)
    if not match_id:
        raise KeyError(
            "`id` not found. Please provide the id of the match in the dictionary."
        )

    home_team = match_info.get("h").get("title")
    away_team = match_info.get("a").get("title")
    date = match_info.get("datetime")

    response = requests.get(MATCH_URL.format(match_id))
    if response.ok:
        soup = BeautifulSoup(response.text, features="lxml")
    else:
        raise RuntimeError(
            f"Could not reach match at understat.com: {response.status}"
        )

    timeline = soup.find_all(
        "div", attrs={"class": "timiline-container"}, recursive=True
    )
    goals = []
    subs = []
    for event in timeline:
        if event.find("i", attrs={"title": "Goal"}):
            scorer = event.find("a", attrs={"class": "player-name"}).text
            goal_time = event.find(
                "span", attrs={"class": "minute-value"}
            ).text[:-1]
            goals.append((scorer, goal_time))
        else:
            row = event.find("div", attrs={"class": "timeline-row"})
            if row.find("i", attrs={"class": "player-substitution"}):
                sub_info = [a.text for a in row.find_all("a")]
                sub_time = event.find(
                    "span", attrs={"class": "minute-value"}
                ).text[:-1]
                sub_info.append(sub_time)
                subs.append(sub_info)

    result = {
        "datetime": date,
        "home": home_team,
        "away": away_team,
        "goals": goals,
        "subs": subs,
    }
    return result


def get_season_info(season: str):
    """Get statistics for whole season

    This function scrapes data for all the matches and returns a single
    dictionary that contains information regarding the goals and
    substitutions for all matches.

    Parameters
    ----------
    season: str
        The season for which the statistics need to be
        reported.


    Returns
    -------
    dict
        Contains all the information regarding the home team,
        the away team, the goals scored and their times, and the
        substitutions made in the match.
    """

    matches_info = get_matches_info(season)
    result = {}

    for match in tqdm(matches_info):
        result[match.get("id")] = parse_match(match)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape understat archives")
    parser.add_argument(
        "--season",
        help="Season to scrape data for",
        choices=["1516", "1617", "1718", "1819", "1920", "2021"],
        required=True,
    )
    args = parser.parse_args()
    season = args.season
    goal_subs_data = get_season_info(season)

    json.dump(
        goal_subs_data,
        open(f"data/goals_subs_data_{season}.json", "w"),
        indent=4,
    )
