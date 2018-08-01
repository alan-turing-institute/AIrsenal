#!/usr/bin/env python

"""
quick'n'dirty script to parse text cut'n'pasted off the FPL site,
and put into a csv file.  Needs 'dateparser' package.
"""

import os
import re
import dateparser

infile = open("../data/gameweeks.txt")
outfile = open("../data/fixtures.csv","w")

outfile.write("gameweek,date,home_team,away_team\n")

fixture_regex = re.compile("([\w\s]+[\w])[\s]+([\d]{2}\:[\d]{2})([\w\s]+[\w])")

gameweek = ""
date = ""
home_team = ""
away_team = ""
date_str = ""
for line in infile.readlines():
    if re.search("Gameweek ([\d]+)",line):
        gameweek = re.search("Gameweek ([\d]+)",line).groups()[0]
        print("gameweek {}".format(gameweek))
    elif re.search("day [\d]+ [A-Z]",line):
        date_str = line.strip()
        date_str += " 2018 "
        print("date {}".format(date_str))
    elif fixture_regex.search(line):
        home_team, ko_time, away_team = fixture_regex.search(line).groups()
        match_time = date_str + ko_time
        date = dateparser.parse(match_time)
        print("{} vs {} {}".format(home_team, away_team, match_time))
        outfile.write("{},{},{},{}\n".format(gameweek,str(date),home_team,away_team))


outfile.close()
