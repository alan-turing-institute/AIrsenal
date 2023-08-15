"""
Functions to help fill the Transaction table, where players are bought and sold,
hopefully with the correct price.  Needs FPL_TEAM_ID to be set, either via environment
variable, or a file named FPL_TEAM_ID in airsenal/data/
"""
from sqlalchemy import and_, or_

from airsenal.framework.schema import Transaction
from airsenal.framework.utils import (
    CURRENT_SEASON,
    NEXT_GAMEWEEK,
    fetcher,
    get_entry_start_gameweek,
    get_player,
    get_player_from_api_id,
    get_players_for_gameweek,
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


def count_transactions(season, fpl_team_id, dbsession=session):
    """Count the number of transactions we have in the database for a given team ID
    and season.
    """
    if fpl_team_id is None:
        fpl_team_id = fetcher.FPL_TEAM_ID

    transactions = (
        dbsession.query(Transaction)
        .filter_by(fpl_team_id=fpl_team_id)
        .filter_by(season=season)
        .all()
    )
    return len(transactions)


def transaction_exists(
    fpl_team_id,
    gameweek,
    season,
    time,
    pid_out,
    price_out,
    pid_in,
    price_in,
    dbsession=session,
):
    """Check whether the transactions related to transferring a player in and out
    in a gameweek at a specific time already exist in the database.
    """
    transactions = (
        dbsession.query(Transaction)
        .filter_by(fpl_team_id=fpl_team_id)
        .filter_by(gameweek=gameweek)
        .filter_by(season=season)
        .filter_by(time=time)
        .filter(
            or_(
                and_(
                    Transaction.player_id == pid_in,
                    Transaction.price == price_in,
                    Transaction.bought_or_sold == 1,
                ),
                and_(
                    Transaction.player_id == pid_out,
                    Transaction.price == price_out,
                    Transaction.bought_or_sold == -1,
                ),
            )
        )
        .all()
    )
    if len(transactions) == 2:  # row for player bought and player sold
        return True
    elif len(transactions) == 0:
        return False
    else:
        raise ValueError(
            f"Database error: {len(transactions)} transactions in the database with "
            f"parameters:  fpl_team_id={fpl_team_id}, gameweek={gameweek}, "
            f"time={time}, pid_in={pid_in}, pid_out={pid_out}. Should be 2."
        )


def add_transaction(
    player_id,
    gameweek,
    in_or_out,
    price,
    season,
    tag,
    free_hit,
    fpl_team_id,
    time,
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
        time=time,
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
        (
            "Getting initially selected players "
            f"in squad {fpl_team_id} for first gameweek..."
        )
    )
    if NEXT_GAMEWEEK == 1:
        print("Season hasn't started yet so nothing to add to the DB.")
        return

    starting_gw = get_entry_start_gameweek(fpl_team_id)
    print(f"Got starting squad from gameweek {starting_gw}.")
    if starting_gw == NEXT_GAMEWEEK:
        print(
            "This is team {fpl_team_id}'s first gameweek so nothing to add to the DB "
            "yet."
        )
        return

    print("Adding player data...")

    init_players = get_players_for_gameweek(starting_gw, fpl_team_id)
    free_hit = free_hit_used_in_gameweek(starting_gw, fpl_team_id)
    time = fetcher.get_event_data()[starting_gw]["deadline"]
    for pid in init_players:
        player_api_id = get_player(pid).fpl_api_id
        first_gw_data = fetcher.get_gameweek_data_for_player(player_api_id, starting_gw)

        if len(first_gw_data) == 0:
            # Edge case where API doesn't have player data for gameweek 1, e.g. in 20/21
            # season where 4 teams didn't play gameweek 1. Calculate GW1 price from
            # API using current price and total price change.
            print(
                (
                    "Using current data to determine "
                    f"starting price for player {player_api_id}"
                )
            )
            pdata = fetcher.get_player_summary_data()[player_api_id]
            price = pdata["now_cost"] - pdata["cost_change_start"]
        else:
            price = first_gw_data[0]["value"]

        add_transaction(
            pid,
            starting_gw,
            1,
            price,
            season,
            tag,
            free_hit,
            fpl_team_id,
            time,
            dbsession,
        )


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
    print(f"Updating db with squad with fpl_team_id={fpl_team_id}")
    # do we already have the initial squad for this fpl_team_id?
    existing_transfers = (
        dbsession.query(Transaction)
        .filter_by(fpl_team_id=fpl_team_id)
        .filter_by(season=season)
        .all()
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
        api_pid_in = transfer["element_in"]
        pid_in = get_player_from_api_id(api_pid_in).player_id
        price_in = transfer["element_in_cost"]
        time = transfer["time"]

        if not transaction_exists(
            fpl_team_id,
            gameweek,
            season,
            time,
            pid_out,
            price_out,
            pid_in,
            price_in,
            dbsession=dbsession,
        ):
            if verbose:
                print(
                    (
                        f"Adding transaction: gameweek: {gameweek} "
                        f"removing player {pid_out} for {price_out}"
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
                time,
                dbsession,
            )

            if verbose:
                print(
                    (
                        f"Adding transaction: gameweek: {gameweek} "
                        f"adding player {pid_in} for {price_in}"
                    )
                )
            add_transaction(
                pid_in,
                gameweek,
                1,
                price_in,
                season,
                tag,
                free_hit,
                fpl_team_id,
                time,
                dbsession,
            )
