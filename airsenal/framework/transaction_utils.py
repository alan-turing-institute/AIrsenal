"""
Functions to help fill the Transaction table, where players are bought and sold,
hopefully with the correct price.  Needs FPL_TEAM_ID to be set, either via environment
variable, or a file named FPL_TEAM_ID in airsenal/data/
"""

from airsenal.framework.schema import Transaction
from airsenal.framework.utils import (
    get_players_for_gameweek,
    fetcher,
    get_player_from_api_id,
    NEXT_GAMEWEEK,
    CURRENT_SEASON,
    get_player,
    session,
)


def free_hit_used_in_gameweek(gameweek, fpl_team_id=None):
    """Use FPL API to determine whether a chip was played in the given gameweek"""
    if not fpl_team_id:
        fpl_team_id = fetcher.FPL_TEAM_ID
    fpl_team_data = fetcher.get_fpl_team_data(gameweek, fpl_team_id)
    if (
        fpl_team_data
        and "active_chip" in fpl_team_data.keys()
        and fpl_team_data["active_chip"] == "freehit"
    ):
        return 1
    else:
        return 0


def add_transaction(
    player_id,
    gameweek,
    in_or_out,
    price,
    season,
    tag,
    free_hit,
    fpl_team_id,
    dbsession=session,
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
        fpl_team_id=fpl_team_id,
    )
    dbsession.add(t)
    dbsession.commit()


def fill_initial_squad(
    season=CURRENT_SEASON,
    tag="AIrsenal" + CURRENT_SEASON,
    fpl_team_id=None,
    dbsession=session,
):
    """
    Fill the Transactions table in the database with the initial 15 players, and their
    costs, getting the information from the team history API endpoint (for the list of
    players in our team) and the player history API endpoint (for their price in gw1).
    """

    if not fpl_team_id:
        fpl_team_id = fetcher.FPL_TEAM_ID
    print(
        "Getting initially selected players in squad {} for first gameweek...".format(
            fpl_team_id
        )
    if NEXT_GAMEWEEK == 1:
        # Season hasn't started yet - there won't be a team in the DB
        return True

    init_players = []
    starting_gw = 0
    while len(init_players) == 0 and starting_gw < NEXT_GAMEWEEK:
        starting_gw += 1
        print(f"Trying gameweek {starting_gw}...")
        init_players = get_players_for_gameweek(starting_gw, fpl_team_id)
        free_hit = free_hit_used_in_gameweek(starting_gw, fpl_team_id)
    print(f"Got starting squad from gameweek {starting_gw}. Adding player data...")
    for pid in init_players:
        player_api_id = get_player(pid).fpl_api_id
        first_gw_data = fetcher.get_gameweek_data_for_player(player_api_id, starting_gw)

        if len(first_gw_data) == 0:
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
            price = first_gw_data[0]["value"]

        add_transaction(pid, 1, 1, price, season, tag, free_hit, fpl_team_id, dbsession)


def update_squad(
    season=CURRENT_SEASON,
    tag="AIrsenal" + CURRENT_SEASON,
    fpl_team_id=None,
    dbsession=session,
    verbose=True,
):
    """
    Fill the Transactions table in the DB with all the transfers in gameweeks after 1,
    using the transfers API endpoint which has the correct buy and sell prices.
    """
    if not fpl_team_id:
        fpl_team_id = fetcher.FPL_TEAM_ID
    print("Updating db with squad with fpl_team_id={}".format(fpl_team_id))
    # do we already have the initial squad for this fpl_team_id?
    existing_transfers = (
        dbsession.query(Transaction).filter_by(fpl_team_id=fpl_team_id).all()
    )
    if len(existing_transfers) == 0:
        # need to put the initial squad into the db
        fill_initial_squad(
            season=season, tag=tag, fpl_team_id=fpl_team_id, dbsession=dbsession
        )
    # now update with transfers
    transfers = fetcher.get_fpl_transfer_data(fpl_team_id)
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
            pid_out,
            gameweek,
            -1,
            price_out,
            season,
            tag,
            free_hit,
            fpl_team_id,
            dbsession,
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
        add_transaction(
            pid_in, gameweek, 1, price_in, season, tag, free_hit, fpl_team_id, dbsession
        )
        pass
