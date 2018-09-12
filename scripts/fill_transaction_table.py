#!/usr/bin/env python

import os
import sys

sys.path.append("..")

import argparse
import json

from framework.mappings import alternative_team_names, positions

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from framework.schema import Transaction, Base, engine
from framework.data_fetcher import FPLDataFetcher

DBSession = sessionmaker(bind=engine)
session = DBSession()


def add_transaction(player_id, gameweek, in_or_out,
                    output_csv_filename=None):
    """
    add buy transactions to the db table
    """
    t = Transaction(player_id=player_id,
                    gameweek=gameweek,
                    bought_or_sold=in_or_out)
    session.add(t)
    session.commit()
    if output_csv_filename:
        output_csv(output_csv_filename,player_id, gameweek, in_or_out)

def output_csv(output_file,player_id,gameweek,in_or_out):
    """
    write out to a csv file
    """
    if not os.path.exists(output_file):
        outfile = open(output_file,"w")
        outfile.write("player_id,gameweek,in_or_out\n")
    else:
        outfile = open(output_file,"a")
    outfile.write("{},{},{}\n".format(player_id,gameweek,in_or_out))
    outfile.close()


def sanity_check_args(args):
    """
    Check we have a consistent set of arguments
    """
    if args.input_csv and args.output_csv:
        print("Can't set both input_csv and output_csv")
        return False
    if args.buy and args.sell:
        print("Can't set buy and sell")
        return False
    if args.buy or args.sell:
        if args.input_csv:
            print("Can't set input_csv and buy or sell")
            return False
        if not args.output_csv:
            print("Need to set output_csv for buy or sell")
            return False
        if not (args.player_id and args.gameweek):
            print("Need to set player_id and gameweek")
            return False
    return True


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Players bought and sold")
    parser.add_argument("--input_csv",help="input CSV file")
    parser.add_argument("--output_csv",help="output CSV file")
    parser.add_argument("--player_id",help="player ID",type=int)
    parser.add_argument("--buy",action='store_true')
    parser.add_argument("--sell",action='store_true')
    parser.add_argument("--gameweek",help="next gameweek after transfer",type=int)
    args = parser.parse_args()
    if not sanity_check_args(args):
        raise RuntimeError("Inconsistent set of arguments")
    if args.output_csv:
        outfile = args.output_csv
    else:
        outfile = None
    if args.input_csv:
        infile = open(args.input_csv)
        for line in infile.readlines()[1:]:
            pid,gw,in_or_out = line.strip().split(",")
            add_transaction(pid,gw,in_or_out,outfile)
    if args.buy:
        add_transaction(args.player_id, args.gameweek,1,outfile)
    if args.sell:
        add_transaction(args.player_id, args.gameweek,-1,outfile)
