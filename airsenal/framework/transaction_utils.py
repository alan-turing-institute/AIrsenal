"""
Functions to help fill the Transaction table, where players are bought and sold,
hopefully with the correct price.  Needs FPL_TEAM_ID to be set, either via environment variable,
or a file named FPL_TEAM_ID in airsenal/data/
"""

from airsenal.framework.schema import Transaction
from airsenal.framework.utils import (
    get_players_for_gameweek,
    fetcher,
    get_player_from_api_id,
    NEXT_GAMEWEEK,
    CURRENT_SEASON,
    get_player,
)


def free_hit_used_in_gameweek(gameweek):
    """Use FPL API to determine whether a chip was played in the given gameweek"""
    if fetcher.get_fpl_team_data(gameweek)["active_chip"] == "freehit":
        return 1
    else:
        return 0


def add_transaction(
    player_id, gameweek, in_or_out, price, season, tag, free_hit, session
):
    """
    add buy (in_or_out=1) or sell (in_or_out=-1) transactions to the db table.
    """
    t = Transaction(
        player_id=player_id,
        gameweek=gameweek,
        bought_or_sold=in_or_out,
        price=price,
        season=season,
        tag=tag,
        free_hit=free_hit,
    )
    session.add(t)
    session.commit()


def fill_initial_team(session, season=CURRENT_SEASON, tag="AIrsenal" + CURRENT_SEASON):
    """
    Fill the Transactions table in the database with the initial 15 players, and their costs,
    getting the information from the team history API endpoint (for the list of players in our team)
    and the player history API endpoint (for their price in gw1).
    """
    print("SQUAD Getting selected players for gameweek 1...")
    if NEXT_GAMEWEEK == 1:
        ### Season hasn't started yet - there won't be a team in the DB
        return True

    free_hit = free_hit_used_in_gameweek(1)
    init_players = get_players_for_gameweek(1)
    for pid in init_players:
        player_api_id = get_player(pid).fpl_api_id
        gw1_data = fetcher.get_gameweek_data_for_player(player_api_id, 1)

        if len(gw1_data) == 0:
            # Edge case where API doesn't have player data for gameweek 1, e.g. in 20/21
            # season where 4 teams didn't play gameweek 1. Calculate GW1 price from
            # API using current price and total price change.
            print(
                "Using current data to determine starting price for player {}".format(
                    player_api_id
                )
            )
            pdata = fetcher.get_player_summary_data()[player_api_id]
            price = pdata["now_cost"] - pdata["cost_change_start"]
        else:
            price = gw1_data[0]["value"]

        add_transaction(pid, 1, 1, price, season, tag, free_hit, session)


def update_team(
    session, season=CURRENT_SEASON, tag="AIrsenal" + CURRENT_SEASON, verbose=True
):
    """
    Fill the Transactions table in the DB with all the transfers in gameweeks after 1, using
    the transfers API endpoint which has the correct buy and sell prices.
    """
    transfers = fetcher.get_fpl_transfer_data()
    for transfer in transfers:
        gameweek = transfer["event"]
        api_pid_out = transfer["element_out"]
        pid_out = get_player_from_api_id(api_pid_out).player_id
        price_out = transfer["element_out_cost"]
        if verbose:
            print(
                "Adding transaction: gameweek: {} removing player {} for {}".format(
                    gameweek, pid_out, price_out
                )
            )
        free_hit = free_hit_used_in_gameweek(gameweek)
        add_transaction(
            pid_out, gameweek, -1, price_out, season, tag, free_hit, session
        )
        api_pid_in = transfer["element_in"]
        pid_in = get_player_from_api_id(api_pid_in).player_id
        price_in = transfer["element_in_cost"]
        if verbose:
            print(
                "Adding transaction: gameweek: {} adding player {} for {}".format(
                    gameweek, pid_in, price_in
                )
            )
        add_transaction(pid_in, gameweek, 1, price_in, season, tag, free_hit, session)
        pass
