"""
quick'n'dirty script to parse text cut'n'pasted off the FPL site,
and put into a csv file.  Needs 'dateparser' package.
"""

import re

import dateparser

from airsenal.framework.output import get_logger

logger = get_logger(__name__)

with (
    open("../data/gameweeks.txt") as infile,
    open("../data/fixtures.csv", "w") as outfile,
):
    outfile.write("gameweek,date,home_team,away_team\n")

    fixture_regex = re.compile(r"([\w\s]+[\w])[\s]+([\d]{2}\:[\d]{2})([\w\s]+[\w])")

    for line in infile.readlines():
        if gws := re.search(r"Gameweek ([\d]+)", line):
            gameweek = gws.groups()[0]
            logger.debug("gameweek %s", gameweek)
        else:
            gameweek = "NA"
        if re.search(r"day [\d]+ [A-Z]", line):
            date_str = line.strip()
            date_str += " 2018 "
            logger.debug("date %s", date_str)
        else:
            date_str = "NA"
        if fixture_details := fixture_regex.search(line):
            home_team, ko_time, away_team = fixture_details.groups()
            match_time = date_str + ko_time
            date = dateparser.parse(match_time)
            logger.debug("%s vs %s %s", home_team, away_team, match_time)
            outfile.write(f"{gameweek},{date!s},{home_team},{away_team}\n")
