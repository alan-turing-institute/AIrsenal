#!/usr/bin/env python

"""
use Selenium and BeautifulSoup to follow selected links and get
information from fplarchives.com.

To setup - need recent version (>67) of Chrome.
brew install selenium
brew cask install chromedriver
pip install bs4
"""

import argparse
import json
import re
import time

from bs4 import BeautifulSoup
from selenium import webdriver

base_url = {
    "1617": "http://fplarchives.com/Seasons/1",
    "1516": "http://fplarchives.com/Seasons/2",
}

browser = webdriver.Chrome()


def parse_detail_page(soup):
    """
    read the week-by-week player data from the parsed html
    return a dict keyed by gameweek.
    """
    player_detail = []
    for row in soup.find_all("tr", attrs={"class": "ng-scope"}):
        gameweek_dict = {
            "gameweek": row.find("td", attrs={"data-ng-bind": "item.gameWeek"}).text
        }
        gameweek_dict["opponent"] = row.find(
            "td", attrs={"data-ng-bind": "item.opponent"}
        ).text
        gameweek_dict["points"] = row.find(
            "td", attrs={"data-ng-bind": "item.points"}
        ).text
        gameweek_dict["goals"] = row.find(
            "td", attrs={"data-ng-bind": "item.goals"}
        ).text
        gameweek_dict["assists"] = row.find(
            "td", attrs={"data-ng-bind": "item.assists"}
        ).text
        gameweek_dict["minutes"] = row.find(
            "td", attrs={"data-ng-bind": "item.minutes"}
        ).text
        gameweek_dict["bonus"] = row.find("td", attrs={"data-ng-bind": "item.bps"}).text
        gameweek_dict["conceded"] = row.find(
            "td", attrs={"data-ng-bind": "item.goalsConceded"}
        ).text
        gameweek_dict["own_goals"] = row.find(
            "td", attrs={"data-ng-bind": "item.ownGoals"}
        ).text
        player_detail.append(gameweek_dict)
    return player_detail


def get_detail_pages(soup, pages_to_go_forward):
    """
    from the (paginated) summary page, loop through the list
    of links to player detail pages, and go to each one, call
    the function to read it, then go back.
    return dict of dicts, with primary key as player name
    We need the pages_to_go_forward arg, as otherwise,
    browser.back() goes back to the first page.
    """
    player_names = []
    player_details_this_page = {}
    for row in soup.find_all("tr", attrs={"class": "ng-scope"}):
        if len(row.find_all("a")) != 1:
            continue
        player_name = row.find("a").text
        player_names.append(player_name)
    # seems a bit crazy to loop again here, but I guess going "back" means
    # the page elements are reset
    print(f"Found {len(player_names)} players on this page")
    for player in player_names:
        print(f"Getting details for {player}")
        player_link = browser.find_element_by_link_text(player)
        player_link.click()
        while len(browser.page_source) < 20000:  # wait for page to load
            time.sleep(1)
        time.sleep(1)
        psource = browser.page_source
        psoup = BeautifulSoup(psource, "lxml")
        player_details_this_page[player] = parse_detail_page(psoup)
        browser.back()
        while len(browser.page_source) < 10000:  # wait for page to load
            time.sleep(1)
        for _ in range(pages_to_go_forward):
            next_button = browser.find_element_by_link_text("Next")
            next_button.click()
            time.sleep(0.1)
    print(f"Returning data for {len(player_details_this_page)} players")
    with open(f"player_details_{pages_to_go_forward}.json", "w") as outfile:
        outfile.write(json.dumps(player_details_this_page))
    return player_details_this_page


def parse_summary_page(soup):
    """
    read the player summary data from the parsed html of one page.
    return a list of dicts, one per player
    """
    summaries = []
    for row in soup.find_all("tr", attrs={"class": "ng-scope"}):
        if len(row.find_all("a")) != 1:
            continue
        player_summary = {"name": row.find("a").text}
        player_summary["team"] = row.find(
            "td", attrs={"data-ng-bind": "item.teamShortName"}
        ).text
        player_summary["position"] = row.find(
            "td", attrs={"data-ng-bind": "item.position"}
        ).text
        player_summary["points"] = row.find(
            "td", attrs={"data-ng-bind": "item.points"}
        ).text
        player_summary["goals"] = row.find(
            "td", attrs={"data-ng-bind": "item.goals"}
        ).text
        player_summary["assists"] = row.find(
            "td", attrs={"data-ng-bind": "item.assists"}
        ).text
        player_summary["minutes"] = row.find(
            "td", attrs={"data-ng-bind": "item.minutes"}
        ).text
        player_summary["reds"] = row.find(
            "td", attrs={"data-ng-bind": "item.reds"}
        ).text
        player_summary["yellows"] = row.find(
            "td", attrs={"data-ng-bind": "item.yellows"}
        ).text
        player_summary["saves"] = row.find(
            "td", attrs={"data-ng-bind": "item.saves"}
        ).text
        player_summary["penalties_saved"] = row.find(
            "td", attrs={"data-ng-bind": "item.pensSaved"}
        ).text
        player_summary["penalties_missed"] = row.find(
            "td", attrs={"data-ng-bind": "item.pensMissed"}
        ).text
        player_summary["bonus"] = row.find(
            "td", attrs={"data-ng-bind": "item.bonus"}
        ).text
        player_summary["clean_sheets"] = row.find(
            "td", attrs={"data-ng-bind": "item.cleanSheets"}
        ).text
        player_summary["cost"] = row.find(
            "td", attrs={"data-ng-show": "seasonId == 1"}
        ).text
        summaries.append(player_summary)
    return summaries


def find_num_players(soup):
    """
    Should be a bit of text on the page saying how many players in total.
    """
    for span in soup.find_all("span", attrs={"class": "items ng-binding"}):
        match = re.search(r"([\d]+) items total", span.text)
        if match:
            return int(match.groups()[0])
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="scrape fplarchives")
    parser.add_argument(
        "--season", help="season - allowed values are '1617' or '1516'", required=True
    )
    parser.add_argument("--mode", help="summary or detailed", default="summary")
    args = parser.parse_args()
    season = args.season
    if season not in ["1516", "1617"]:
        raise RuntimeError("Please specify the season - 1516 or 1617")
    if args.mode == "summary":
        output_file = open(f"player_summary_{season}.json", "w")
        player_data = []
    else:
        output_file = open(f"player_details_{season}.json", "w")
        player_data = {}

    # go to the starting page
    browser.get(base_url[season])
    while len(browser.page_source) < 40000:  # wait for page to load
        time.sleep(1)
    source = browser.page_source
    soup = BeautifulSoup(source, "lxml")
    num_total_players = find_num_players(soup)
    pages_done = 0
    #    # go through the pagination
    while len(player_data) < num_total_players:
        if args.mode == "summary":
            player_data_this_page = parse_summary_page(soup)
            player_data += player_data_this_page
        else:
            player_data_this_page = get_detail_pages(soup, pages_done)
            player_data = {**player_data, **player_data_this_page}
        next_link = browser.find_element_by_link_text("Next")
        next_link.click()
        time.sleep(1)
        source = browser.page_source
        soup = BeautifulSoup(source, "lxml")
        pages_done += 1

    # write to the output
    json_string = json.dumps(player_data)
    output_file.write(json_string)  # .encode("utf-8"))
    output_file.close()
