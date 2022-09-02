#!/usr/bin/env python

"""
quick'n'dirty script to parse text cut'n'pasted off the FPL site,
and put into a csv file.  Needs 'dateparser' package.
"""


import re

import dateparser

infile = open("../data/gameweeks.txt")
with open("../data/fixtures.csv", "w") as outfile:
    outfile.write("gameweek,date,home_team,away_team\n")

    fixture_regex = re.compile(r"([\w\s]+[\w])[\s]+([\d]{2}\:[\d]{2})([\w\s]+[\w])")

    gameweek = ""
    date = ""
    home_team = ""
    away_team = ""
    date_str = ""
    for line in infile.readlines():
        if re.search(r"Gameweek ([\d]+)", line):
            gameweek = re.search(r"Gameweek ([\d]+)", line).groups()[0]
            print(f"gameweek {gameweek}")
        elif re.search(r"day [\d]+ [A-Z]", line):
            date_str = line.strip()
            date_str += " 2018 "
            print(f"date {date_str}")
        elif fixture_regex.search(line):
            home_team, ko_time, away_team = fixture_regex.search(line).groups()
            match_time = date_str + ko_time
            date = dateparser.parse(match_time)
            print(f"{home_team} vs {away_team} {match_time}")
            outfile.write(f"{gameweek},{str(date)},{home_team},{away_team}\n")
