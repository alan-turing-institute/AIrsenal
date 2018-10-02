#!/usr/bin/env python

"""
Fill the "Player" table with info from this seasons FPL
(FPL_2017-18.json).
"""

from airsenal import fill_player_table_from_api, \
    fill_player_table_from_file

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="fill player table")
    parser.add_argument("--input_file",help="input json")
    parser.add_argument("--season",help="1516,1617, or 1819",default="1819")
    parser.add_argument("--use_api",action='store_true')
    args = parser.parse_args()
    if args.use_api and args.input_file:
        raise RuntimeError("Specify just one of input_file or use_api")
    if args.use_api:
        fill_player_table_from_api(args.season)
    elif args.input_file:
        fill_player_table_from_file(args.input_file, args.season)
