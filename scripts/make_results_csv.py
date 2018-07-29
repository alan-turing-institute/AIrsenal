#!/usr/bin/env python

"""
quick'n'dirty python script to convert results copy/pasted from
http://www.skysports.com/premier-league-results/YYYY-YY

Usage example:
python make_results_csv.py 2017
(to get results for 2017-18 season
"""

import sys
import os
import re
from datetime import datetime

date_regex = re.compile("day ([\d]+)[a-z]{2}[\s]+([\w]+)")
score_regex = re.compile("([\d])[\s]+([\d])[\s]+([\w\s]+[\w])[\s]+FT")

start_year = sys.argv[-1]
start_year_short = start_year[-2:]
end_year_short = str(int(start_year_short)+1)
end_year = "20" + end_year_short

infilename = "data/results{}{}.txt".format(start_year_short,end_year_short)
outfilename = "data/results_{}{}.csv".format(start_year_short,end_year_short)

infile = open(infilename)
outfile = open(outfilename,"w")
outfile.write("date,home_team,away_team,home_score,away_score\n")

home_team = ""
away_team = ""
datestr = ""
for line in infile.readlines():
    if date_regex.search(line):
        day, month = date_regex.search(line).groups()
        if month in ["January","February","March","April","May"]:
            year = end_year
        else:
            year = start_year
        date = datetime.strptime("{} {} {}".format(day,month[:3],year),
                                 "%d %b %Y")
        datestr = str(date)
        print("Found date {}".format(date))
        last_line_was_score = False
    elif score_regex.search(line):
        home_score, away_score, away_team = score_regex.search(line).groups()
        outfile.write("{},{},{},{},{}\n".format(datestr,
                                                home_team,
                                                away_team,
                                                home_score,
                                                away_score))
    else:
        home_team = line.strip()

outfile.close()
