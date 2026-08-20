"""
simple script, check whether recent matches have been played since
the last entries in the DB, and update the transactions table with players
bought or sold.
"""

from sqlalchemy.orm.session import Session

from airsenal.framework.output import console, get_logger
from airsenal.framework.schema import Player, database_is_empty, session_scope
from airsenal.framework.transaction_utils import count_transactions, update_squad
from airsenal.framework.utils import (
    NEXT_GAMEWEEK,
    fetcher,
    get_last_complete_gameweek_in_db,
    get_last_finished_gameweek,
    list_players,
)
from airsenal.scripts.fill_fixture_table import fill_fixtures_from_api
from airsenal.scripts.fill_player_attributes_table import fill_attributes_table_from_api
from airsenal.scripts.fill_player_mappings_table import add_mappings
from airsenal.scripts.fill_player_table import find_player_in_table
from airsenal.scripts.fill_playerscore_table import fill_playerscores_from_api
from airsenal.scripts.fill_result_table import fill_results_from_api

logger = get_logger(__name__)


def update_transactions(season: str, fpl_team_id: int, dbsession: Session) -> bool:
    """
    Ensure that the transactions table in the database is up-to-date.
    """
    if NEXT_GAMEWEEK != 1:
        logger.info("Checking team")
        n_transfers_api = len(fetcher.get_fpl_transfer_data(fpl_team_id))
        n_transactions_db = count_transactions(season, fpl_team_id, dbsession)
        # DB has 2 rows per transfer, and rows for the 15 players selected in the
        # initial squad which are not returned by the transfers API
        n_transfers_db = (n_transactions_db - 15) / 2
        if n_transfers_db != n_transfers_api:
            update_squad(
                season=season,
                fpl_team_id=fpl_team_id,
                dbsession=dbsession,
            )
        else:
            logger.info("Team is up-to-date")
    else:
        logger.info("No transactions as season hasn't started")
    return True


def update_results(season: str, dbsession: Session) -> bool:
    """
    If the last gameweek in the db is earlier than the last finished gameweek,
    update the 'results', 'playerscore', and (optionally) 'attributes' tables.
    """
    last_in_db = get_last_complete_gameweek_in_db(season, dbsession=dbsession)
    if not last_in_db:
        # no results in database for this season yet
        last_in_db = 0
    last_finished = get_last_finished_gameweek()

    if NEXT_GAMEWEEK == 1:
        logger.info("Skipping team and result updates - season hasn't started.")
    elif last_finished > last_in_db:
        # need to update
        logger.info("Updating results table ...")
        fill_results_from_api(
            gw_start=last_in_db + 1,
            gw_end=NEXT_GAMEWEEK,
            season=season,
            dbsession=dbsession,
        )
        logger.info("Updating playerscores table ...")
        fill_playerscores_from_api(
            season=season,
            gw_start=last_in_db + 1,
            gw_end=NEXT_GAMEWEEK,
            dbsession=dbsession,
        )
    else:
        logger.info("Matches and player-scores already up-to-date")
    return True


def update_players(season: str, dbsession: Session) -> int:
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
        logger.info("Player table already up-to-date.")
        return 0
    if len(players_from_db) > len(players_from_api):
        msg = "Something strange has happened - more players in DB than API"
        raise RuntimeError(msg)
    return add_players_to_db(
        players_from_db, players_from_api, player_data_from_api, dbsession
    )


def add_players_to_db(
    players_from_db: list,
    players_from_api: list[int],
    player_data_from_api: dict,
    dbsession: Session,
) -> int:
    logger.info("Updating player table...")
    # find the new player(s) from the API
    api_ids_from_db = [p.fpl_api_id for p in players_from_db]
    new_players = [p for p in players_from_api if p not in api_ids_from_db]
    for player_api_id in new_players:
        first_name = player_data_from_api[player_api_id]["first_name"]
        second_name = player_data_from_api[player_api_id]["second_name"]
        name = f"{first_name} {second_name}"
        # check whether we already have this player in the database -
        # if yes update that player's data, if no create a new player
        p = find_player_in_table(name, dbsession=dbsession)
        if p is None:
            logger.info("Adding player %s", name)
            p = Player()
            update = False
        elif p.fpl_api_id is None:
            logger.info("Updating player %s", name)
            update = True
        else:
            update = True
        p.fpl_api_id = player_api_id
        p.name = name
        if not update:
            dbsession.add(p)
            add_mappings(p, dbsession=dbsession)

    dbsession.commit()
    return len(new_players)


def update_attributes(season: str, dbsession: Session) -> None:
    """Update player attributes table"""
    # update from, and including, the last gameweek we have results for in the
    # database (including that gameweek as player prices etc. can change after
    # matches have finished but before the next gameweek deadline)
    last_in_db = get_last_complete_gameweek_in_db(season, dbsession=dbsession)
    if not last_in_db:
        # no results in database for this season yet
        last_in_db = 0

    logger.info("Updating attributes table ...")
    fill_attributes_table_from_api(
        season=season,
        gw_start=last_in_db,
        dbsession=dbsession,
    )


def update_db(
    season: str, do_attributes: bool, fpl_team_id: int, session: Session
) -> bool:
    with console.status("Updating the database..."):
        # see if any new players have been added
        num_new_players = update_players(season, session)

        # update player attributes (if requested)
        if not do_attributes and num_new_players > 0:
            logger.info("New players added - enforcing update of attributes table")
            do_attributes = True
        if do_attributes:
            update_attributes(season, session)

        # update fixtures (which may have been rescheduled)
        logger.info("Updating fixture table...")
        fill_fixtures_from_api(season, session)
        # update results and playerscores
        update_results(season, session)
        # update our squad
        update_transactions(season, fpl_team_id, session)
    return True


def update_database(season: str, noattr: bool, fpl_team_id: int | None) -> None:
    """Update database tables from current FPL data."""
    do_attributes = not noattr
    fpl_team_id = fpl_team_id or fetcher.FPL_TEAM_ID
    if not fpl_team_id:
        msg = "FPL team ID must be specified in args, config, or env"
        raise ValueError(msg)

    with session_scope() as session:
        if database_is_empty(session):
            logger.warning("Database is empty, run 'airsenal_setup_initial_db' first")
            return

        update_db(season, do_attributes, fpl_team_id, session)
