#!/usr/bin/env python

import os
import argparse

from ..framework.schema import Transaction, session_scope
from ..framework.utils import get_players_for_gameweek, fetcher



def add_transaction(player_id, gameweek, in_or_out, price, season, tag, session):
    """
    add buy (in_or_out=1) or sell (in_or_out=-1) transactions to the db table.
    """
    t = Transaction(player_id=player_id, gameweek=gameweek, bought_or_sold=in_or_out, price=price, season=season, tag=tag)
    session.add(t)
    session.commit()


def fill_initial_team(session, season="1819", tag="AIrsenal1819"):
    """
    Fill the Transactions table in the database with the initial 15 players, and their costs,
    getting the information from the team history API endpoint (for the list of players in our team)
    and the player history API endpoint (for their price in gw1).
    """
    api_players = get_players_for_gameweek(1)
    for pid in api_players:
        gw1_data = fetcher.get_gameweek_data_for_player(pid,1)
        price = gw1_data[0]['value']
        add_transaction(pid, 1, 1, price, season, tag, session)


def update_team(session, season="1819", tag="AIrsenal1819"):
    """
    Fill the Transactions table in the DB with all the transfers in gameweeks after 1, using
    the transfers API endpoint which has the correct buy and sell prices.
    """
    transfers = fetcher.get_fpl_transfer_data()
    for transfer in transfers:
        gameweek = transfer['event']
        pid_out = transfer['element_out']
        price_out = transfer['element_out_cost']
        add_transaction(pid_out, gameweek, -1, price_out, season, tag, session)
        pid_in = transfer['element_in']
        price_in = transfer['element_in_cost']
        add_transaction(pid_in, gameweek, 1, price_in, season, tag, session)
        pass



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Players bought and sold")
    parser.add_argument("--player_id", help="player ID", type=int)
    parser.add_argument("--buy", action="store_true")
    parser.add_argument("--sell", action="store_true")
    parser.add_argument("--price", type=int, help="price in 0.1Millions")
    parser.add_argument("--gameweek", help="next gameweek after transfer", type=int)
    parser.add_argument("--tag", help="identifying tag", default="AIrsenal1819")
    parser.add_argument("--season", help="which season, in format e.g. '1819'",default="1819")
    args = parser.parse_args()

    with session_scope() as session:
        make_transaction_table(session, args=args)
