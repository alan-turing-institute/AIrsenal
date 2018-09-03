#!/usr/bin/env python

import sys
sys.path.append("..")

import argparse
import json

from framework.mappings import alternative_team_names, positions

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from framework.schema import Transaction, Base, engine
from framework.data_fetcher import DataFetcher

DBSession = sessionmaker(bind=engine)
session = DBSession()


def buy_player(player_id, gameweek):
    """
    add buy transactions to the db table
    """
    t = Transaction(player_id=player_id,
                    gameweek=gameweek,
                    bought_or_sold=1)
    session.add(t)
    pass

def sell_player(player_id, gameweek):
    """
    add sell transactions to the db table
    """
    t = Transaction(player_id=player_id,
                    gameweek=gameweek,
                    bought_or_sold=-1)
    session.add(t)

if __name__ == "__main__":
    ###### initial team
    ## Vorm (352) Begovic (24)
    ## Alexander-Arnold (245) Azpilicueta (113) Alonso (115) Kelly (140) Wan-Bissake (145)
    ## Son (367) Salah (253) D.Silva (271) Fabregas (123) Moura (370)
    ## King (45) Mousset (44) Firmino (257)
    parser = argparse.ArgumentParser(description="History of the AIrsenal")
    parser.add_argument("--gw_start",help="start_gw",type=int,default=0)
    parser.add_argument("--gw_end",help="end_gw",type=int,default=99)
    args = parser.parse_args()

    if 1 in range(args.gw_start,args.gw_end):
        for pid in [352,24,245,113,115,140,145,367,253,271,123,370,45,44,257]:
            buy_player(pid, 1)

    ##### gw 2, sold Son, bought Bernardo Silva (276)
    if 2 in range(args.gw_start,args.gw_end):
        sell_player(367,2)
        buy_player(276,2)

    ##### gw 3, sold Fabregas, bought Pedro (125)
    if 3 in range(args.gw_start,args.gw_end):
        sell_player(123,3)
        buy_player(125,3)

    ##### gw 4, sold David Silva, bought Walcott (164)
    if 4 in range(args.gw_start,args.gw_end):
        sell_player(271,4)
        buy_player(164,4)

    session.commit()
