#!/usr/bin/env python

"""
Use BeautifulSoup to follow links to scrape data from understat.com

To setup -
`pip install bs4`
"""

import argparse
import json
import os
from datetime import datetime, timedelta

import pytz
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

LEAGUE_URL = "https://understat.com/league/epl/{}"
MATCH_URL = "https://understat.com/match/{}"
base_url = {
    "1516": LEAGUE_URL.format("2015"),
    "1617": LEAGUE_URL.format("2016"),
    "1718": LEAGUE_URL.format("2017"),
    "1819": LEAGUE_URL.format("2018"),
    "1920": LEAGUE_URL.format("2019"),
    "2021": LEAGUE_URL.format("2020"),
    "2122": LEAGUE_URL.format("2021"),
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
    try:
        response = requests.get(base_url[season])
    except KeyError:
        raise KeyError(
            f"Please provide valid season to scrape data: "
            f"{season} not in {list(base_url.keys())}"
        )

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
            f"Could not receive data for the given season. "
            f"Error code: {response.status_code}"
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
            "goals": (dict) goals,
            "subs": (dict) subs,
        }
        The `goals` dict is structured as a dictinonary of lists
        {"home": [], "away": []} where each entry of the list is of the
        form [goal_scorer, time_of_goal]. The subs dict is also a dict of
        lists of the form {"home": [], "away": []} with each entry of the
        form [player_in, player_out, time_of_substitution].
    """
    match_id = match_info.get("id", None)
    if not match_id:
        raise KeyError(
            "`id` not found. Please provide the id of the match in the dictionary."
        )

    home_team = match_info.get("h").get("title")
    away_team = match_info.get("a").get("title")
    date = match_info.get("datetime")
    if not date or datetime.fromisoformat(date) + timedelta(hours=3) > datetime.now(
        pytz.utc
    ):
        # match not played yet or only recently finished, skip
        return None

    response = requests.get(MATCH_URL.format(match_id))
    if response.ok:
        soup = BeautifulSoup(response.text, features="lxml")
    else:
        raise RuntimeError(
            f"Could not reach match {match_id} "
            f"({home_team} vs. {away_team}, {date}) "
            f"at understat.com: {response.status_code}"
        )

    timeline = soup.find_all(
        "div", attrs={"class": "timiline-container"}, recursive=True
    )
    goals = {"home": [], "away": []}
    subs = {"home": [], "away": []}
    for event in timeline:
        if event.find("i", attrs={"class": "fa-futbol"}):
            scorer = event.find("a", attrs={"class": "player-name"}).text
            goal_time = event.find("span", attrs={"class": "minute-value"}).text[:-1]
            block = event.find("div", attrs={"class": "timeline-row"}).find_parent()
            if "block-home" in block["class"]:
                goals["home"].append((scorer, goal_time))
            else:
                goals["away"].append((scorer, goal_time))
        rows = event.find_all("div", attrs={"class": "timeline-row"})
        if rows:
            for r in rows:
                if r.find("i", attrs={"class": "player-substitution"}):
                    sub_info = [a.text for a in r.find_all("a")]
                    sub_time = event.find("span", attrs={"class": "minute-value"}).text[
                        :-1
                    ]
                    sub_info.append(sub_time)

                    block = r.find_parent()
                    if "block-home" in block["class"]:
                        subs["home"].append(sub_info)
                    else:
                        subs["away"].append(sub_info)

    result = {
        "datetime": date,
        "home": home_team,
        "away": away_team,
        "goals": goals,
        "subs": subs,
    }
    return result


def get_season_info(season: str, result: dict = {}):
    """Get statistics for whole season

    This function scrapes data for all the matches and returns a single
    dictionary that contains information regarding the goals and
    substitutions for all matches.

    Parameters
    ----------
    season: str
        The season for which the statistics need to be
        reported.
    results: dict, optional
        Previously saved match results - won't get new data for any match ID present
        in results.keys(), by default {}

    Returns
    -------
    dict
        Contains all the information regarding the home team,
        the away team, the goals scored and their times, and the
        substitutions made in the match.
    """

    matches_info = get_matches_info(season)

    for match in tqdm(matches_info):
        if match.get("id") not in result.keys():
            parsed_match = parse_match(match)
            if parsed_match:
                result[match.get("id")] = parsed_match

    return result


def main():
    parser = argparse.ArgumentParser(description="Scrape understat archives")
    parser.add_argument(
        "--season",
        help="Season to scrape data for",
        choices=list(base_url.keys()),
        required=True,
    )
    parser.add_argument(
        "--overwrite",
        help="Force overwriting previously saved data if set",
        action="store_true",
    )
    args = parser.parse_args()
    season = args.season
    overwrite = args.overwrite

    result = {}
    save_path = os.path.join(
        os.path.dirname(__file__), f"../data/goals_subs_data_{season}.json"
    )
    if os.path.exists(save_path) and not overwrite:
        print(
            f"Data for {season} season already exists. Will only get data for new "
            "matches. To re-download data for all matches use --overwrite."
        )
        with open(save_path, "r") as f:
            result = json.load(f)

    goal_subs_data = get_season_info(season, result=result)

    with open(save_path, "w") as f:
        json.dump(goal_subs_data, f, indent=4)


if __name__ == "__main__":
    main()
