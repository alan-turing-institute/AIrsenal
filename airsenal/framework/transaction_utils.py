"""
Functions to help fill the Transaction table, where players are bought and sold,
hopefully with the correct price.  Needs FPL_TEAM_ID to be set, either via environment variable,
or a file named FPL_TEAM_ID in airsenal/data/
"""

import os

from .schema import Transaction, session_scope
from .utils import get_players_for_gameweek, fetcher, \
    get_past_seasons, NEXT_GAMEWEEK, CURRENT_SEASON



def add_transaction(player_id, gameweek, in_or_out, price, season, tag, session):
    """
    add buy (in_or_out=1) or sell (in_or_out=-1) transactions to the db table.
    """
    t = Transaction(player_id=player_id, gameweek=gameweek, bought_or_sold=in_or_out, price=price, season=season, tag=tag)
    session.add(t)
    session.commit()


def fill_initial_team(session, season=CURRENT_SEASON, tag="AIrsenal"+CURRENT_SEASON):
    """
    Fill the Transactions table in the database with the initial 15 players, and their costs,
    getting the information from the team history API endpoint (for the list of players in our team)
    and the player history API endpoint (for their price in gw1).
    """
    if NEXT_GAMEWEEK == 1:
        ### Season hasn't started yet - there won't be a team in the DB
        return True
    api_players = get_players_for_gameweek(1)
    for pid in api_players:
        gw1_data = fetcher.get_gameweek_data_for_player(pid,1)
        price = gw1_data[0]['value']
        add_transaction(pid, 1, 1, price, season, tag, session)


def update_team(session, season=CURRENT_SEASON, tag="AIrsenal"+CURRENT_SEASON, verbose=True):
    """
    Fill the Transactions table in the DB with all the transfers in gameweeks after 1, using
    the transfers API endpoint which has the correct buy and sell prices.
    """
    transfers = fetcher.get_fpl_transfer_data()
    for transfer in transfers:
        gameweek = transfer['event']
        pid_out = transfer['element_out']
        price_out = transfer['element_out_cost']
        if verbose:
            print("Adding transaction: gameweek: {} removing player {} for {}"\
                  .format(gameweek, pid_out, price_out))
        add_transaction(pid_out, gameweek, -1, price_out, season, tag, session)
        pid_in = transfer['element_in']
        price_in = transfer['element_in_cost']
        if verbose:
            print("Adding transaction: gameweek: {} adding player {} for {}"\
                  .format(gameweek, pid_in, price_in))
        add_transaction(pid_in, gameweek, 1, price_in, season, tag, session)
        pass
