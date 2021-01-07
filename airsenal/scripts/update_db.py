#!/usr/bin/env python

"""
simple script, check whether recent matches have been played since
the last entries in the DB, and update the transactions table with players
bought or sold.
"""
import argparse

from airsenal.framework.utils import (
    CURRENT_SEASON,
    get_last_gameweek_in_db,
    get_last_finished_gameweek,
    NEXT_GAMEWEEK,
    get_current_players,
    get_players_for_gameweek,
    list_players,
    fetcher,
    get_player,
)
from airsenal.scripts.fill_player_attributes_table import fill_attributes_table_from_api
from airsenal.scripts.fill_result_table import fill_results_from_api
from airsenal.scripts.fill_playerscore_table import fill_playerscores_from_api
from airsenal.framework.transaction_utils import update_squad
from airsenal.framework.schema import Player, session_scope


def update_transactions(season, dbsession):
    """
    Ensure that the transactions table in the database is up-to-date.
    """

    if NEXT_GAMEWEEK != 1:
        print("Checking team")
        current_gameweek = NEXT_GAMEWEEK - 1
        db_players = sorted(get_current_players(season=season, dbsession=dbsession))
        api_players = sorted(get_players_for_gameweek(current_gameweek))
        if db_players != api_players:
            update_squad(season=season, dbsession=dbsession, verbose=True)
        else:
            print("Team is up-to-date")
    else:
        print("No transactions as season hasn't yet started")
    return True


def update_results(season, dbsession):
    """
    If the last gameweek in the db is earlier than the last finished gameweek,
    update the 'results', 'playerscore', and (optionally) 'attributes' tables.
    """
    last_in_db = get_last_gameweek_in_db(season, dbsession=dbsession)
    if not last_in_db:
        # no results in database for this season yet
        last_in_db = 0
    last_finished = get_last_finished_gameweek()

    if NEXT_GAMEWEEK != 1:
        if last_finished > last_in_db:
            # need to update
            print("Updating results table ...")
            fill_results_from_api(
                last_in_db + 1, NEXT_GAMEWEEK, season, dbsession=dbsession
            )
            print("Updating playerscores table ...")
            fill_playerscores_from_api(season, dbsession, last_in_db + 1, NEXT_GAMEWEEK)
        else:
            print("Matches and player-scores already up-to-date")
    else:
        print("Skipping team and result updates - season hasn't started.")
    return True


def update_players(season, dbsession):
    """
    See if any new players have been added to FPL since we last filled the 'player'
    table in the db.  If so, add them.
    """
    players_from_db = list_players(
        position="all", team="all", season=season, dbsession=dbsession
    )
    player_data_from_api = fetcher.get_player_summary_data()
    players_from_api = list(player_data_from_api.keys())

    if len(players_from_db) == len(players_from_api):
        print("Player table already up-to-date.")
        return 0
    elif len(players_from_db) > len(players_from_api):
        raise RuntimeError(
            "Something strange has happened - more players in DB than API"
        )
    else:
        # find the new player(s) from the API
        api_ids_from_db = [p.fpl_api_id for p in players_from_db]
        new_players = [p for p in players_from_api if p not in api_ids_from_db]
        for player_api_id in new_players:
            first_name = player_data_from_api[player_api_id]["first_name"]
            second_name = player_data_from_api[player_api_id]["second_name"]
            name = "{} {}".format(first_name, second_name)
            print("Adding player {}".format(name))
            # check whether we alreeady have this player in the database -
            # if yes update that player's data, if no create a new player
            p = get_player(name, dbsession=dbsession)
            if p is None:
                p = Player()
                update = False
            else:
                update = True
            p.fpl_api_id = player_api_id
            p.name = name
            if not update:
                dbsession.add(p)
        dbsession.commit()
        return len(new_players)


def update_attributes(season, dbsession):
    """Update player attributes table"""
    # update from, and including, the last gameweek we have results for in the
    # database (including that gameweek as player prices etc. can change after
    # matches have finished but before the next gameweek deadline)
    last_in_db = get_last_gameweek_in_db(season, dbsession=dbsession)
    if not last_in_db:
        # no results in database for this season yet
        last_in_db = 0

    print("Updating attributes table ...")
    fill_attributes_table_from_api(
        season=season,
        gw_start=last_in_db,
        gw_end=NEXT_GAMEWEEK,
        dbsession=dbsession,
    )


def main():

    parser = argparse.ArgumentParser(
        description="fill db tables with recent scores and transactions"
    )
    parser.add_argument(
        "--season", help="season, in format e.g. '1819'", default=CURRENT_SEASON
    )
    parser.add_argument(
        "--noattr", help="don't update player attributes", action="store_true"
    )
    parser.add_argument(
        "--fpl_team_id",
        help="specify fpl team id",
        required=False,
        default=None,
    )

    args = parser.parse_args()

    season = args.season
    do_attributes = not args.noattr
    if args.fpl_team_id is not None:
        fetcher.FPL_TEAM_ID = args.fpl_team_id

    with session_scope() as session:
        # see if any new players have been added
        num_new_players = update_players(season, session)

        # update player attributes (if requested)
        if not do_attributes and num_new_players > 0:
            print("New players added - enforcing update of attributes table")
            do_attributes = True

        if do_attributes:
            update_attributes(season, session)

        # update results and playerscores
        update_results(season, session)
        # update our squad
        update_transactions(season, session)


# TODO update fixtures table (e.g. in case of rescheduling)?


if __name__ == "__main__":
    print(" ==== updating results and transactions === ")
    main()
