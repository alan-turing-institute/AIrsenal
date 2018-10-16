#!/usr/bin/env python

import os
import argparse

from ..framework.schema import Transaction, session_scope

from collections import namedtuple

TransferArgs = namedtuple("TransferArgs",
                          ["input_csv", "output_csv", "player_id", "buy", "sell", "gameweek","tag"])


def add_transaction(player_id, gameweek, in_or_out, season, tag, session, output_csv_filename=None):
    """
    add buy transactions to the db table
    """
    t = Transaction(player_id=player_id, gameweek=gameweek, bought_or_sold=in_or_out, season=season, tag=tag)
    session.add(t)
    session.commit()
    if output_csv_filename:
        output_csv(output_csv_filename, player_id, gameweek, in_or_out, season, tag)


def output_csv(output_file, player_id, gameweek, in_or_out, season, tag):
    """
    write out to a csv file
    """
    if not os.path.exists(output_file):
        outfile = open(output_file, "w")
        outfile.write("player_id,gameweek,in_or_out,season,tag\n")
    else:
        outfile = open(output_file, "a")
    outfile.write("{},{},{},{},{}\n".format(player_id, gameweek, in_or_out, season, tag))
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


def make_transaction_table(session,
                           season="1819",
                           args=TransferArgs(os.path.join(os.path.dirname(__file__), "../data/transactions.csv"),
                                             None,
                                             None,
                                             None,
                                             None,
                                             None,
                                             None)):

    if not sanity_check_args(args):
        raise RuntimeError("Inconsistent set of arguments")
    if args.output_csv:
        outfile = args.output_csv
    else:
        outfile = None
    if args.input_csv:
        infile = open(args.input_csv)
        for line in infile.readlines()[1:]:
            pid, gw, in_or_out,tag = line.strip().split(",")
            add_transaction(pid, gw, in_or_out, season, tag, session, output_csv_filename=outfile)
    if args.buy:
        add_transaction(args.player_id, args.gameweek, 1, season, args.tag, session, output_csv_filename=outfile)
    if args.sell:
        add_transaction(args.player_id, args.gameweek, -1, season, args.tag, session, output_csv_filename=outfile)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Players bought and sold")
    parser.add_argument("--input_csv",
                        help="input CSV file",
                        default=os.path.join(os.path.dirname(__file__), "../data/transactions.csv"))
    parser.add_argument("--output_csv", help="output CSV file")
    parser.add_argument("--player_id", help="player ID", type=int)
    parser.add_argument("--buy", action="store_true")
    parser.add_argument("--sell", action="store_true")
    parser.add_argument("--gameweek", help="next gameweek after transfer", type=int)
    parser.add_argument("--season", help="which season, in format e.g. '1819'",default="1819")
    args = parser.parse_args()

    with session_scope() as session:
        make_transaction_table(session, args=args)
