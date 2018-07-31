#!/usr/bin/env python

"""
use Selenium and BeautifulSoup to follow selected links and get
information from fplarchives.com.

To setup - need recent version (>67) of Chrome.
brew install selenium
brew cask install chromedriver
pip install bs4
"""

import sys
import time
import json
import re

from selenium import webdriver
from bs4 import BeautifulSoup

base_url = {"1617": "http://fplarchives.com/Seasons/1",
            "1516":"http://fplarchives.com/Seasons/2"
}

browser = webdriver.Chrome()

def has_next_page_link(soup):
    """
    return True if the page has a pagination link "Next" -
    if it doesn't we're presumably at the last page.
    """
    for link in soup.find_all("a"):
        if link.text == "Next":
            return True
    return False

def parse_summary_page(soup):
    """
    read the player summary data from the parsed html of one page.
    return a list of dicts, one per player
    """
    summaries = []
    for row in soup.find_all("tr",attrs={"class":"ng-scope"}):
        if len(row.find_all("a")) != 1:
            continue
        player_summary = {}
        player_summary["name"] = row.find("a").text
        player_summary["team"] = row.find("td",attrs={"data-ng-bind":"item.teamShortName"}).text
        player_summary["position"] = row.find("td",attrs={"data-ng-bind":"item.position"}).text
        player_summary["points"] = row.find("td",attrs={"data-ng-bind":"item.points"}).text
        player_summary["goals"] = row.find("td",attrs={"data-ng-bind":"item.goals"}).text
        player_summary["assists"] = row.find("td",attrs={"data-ng-bind":"item.assists"}).text
        player_summary["minutes"] = row.find("td",attrs={"data-ng-bind":"item.minutes"}).text
        player_summary["reds"] = row.find("td",attrs={"data-ng-bind":"item.reds"}).text
        player_summary["yellows"] = row.find("td",attrs={"data-ng-bind":"item.yellows"}).text
        player_summary["saves"] = row.find("td",attrs={"data-ng-bind":"item.saves"}).text
        player_summary["penalties_saved"] = row.find("td",attrs={"data-ng-bind":"item.pensSaved"}).text
        player_summary["penalties_missed"] = row.find("td",attrs={"data-ng-bind":"item.pensMissed"}).text
        player_summary["bonus"] = row.find("td",attrs={"data-ng-bind":"item.bonus"}).text
        player_summary["clean_sheets"] = row.find("td",attrs={"data-ng-bind":"item.cleanSheets"}).text
        player_summary["cost"] = row.find("td",attrs={"data-ng-show":"seasonId == 1"}).text
        summaries.append(player_summary)
    return summaries

def find_num_players(soup):
    """
    Should be a bit of text on the page saying how many players in total.
    """
    for span in soup.find_all("span",attrs={"class":"items ng-binding"}):
        match =  re.search("([\d]+) items total", span.text)
        if match:
            return int(match.groups()[0])
    return 0


if __name__ == "__main__":
    season = sys.argv[-1]
    if not (season == "1516" or season == "1617"):
        raise RuntimeError("Please specify the season - 1516 or 1617")
    summary_file = open("../data/player_summary_{}.json".format(season),"w")
#    summary_file.write("name,team,position,points,goals,assists,minutes,reds,yellows,saves,PS,PM,bonus,CS,cost\n")
    player_summaries = []
    player_details = []
    # go to the starting page
    browser.get(base_url[season])
    while len(browser.page_source) < 40000: # wait for page to load
        time.sleep(1)
    source = browser.page_source
    soup = BeautifulSoup(source,"lxml")
    num_total_players = find_num_players(soup)
    # go through the pagination
    while has_next_page_link(soup):
        player_summaries_this_page = parse_summary_page(soup)
        player_summaries += player_summaries_this_page
        next_link = browser.find_element_by_link_text("Next")
        next_link.click()
        time.sleep(1)
        old_source = source
        source = browser.page_source
        if len(player_summaries) == num_total_players:
            break
        soup = BeautifulSoup(source,"lxml")
    # now need to do the last page
#    player_summaries_this_page = parse_summary_page(soup)
#    player_summaries += player_summaries_this_page

    json_string = json.dumps(player_summaries)
    summary_file.write(json_string)#.encode("utf-8"))
    summary_file.close()
