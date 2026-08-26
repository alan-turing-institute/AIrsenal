"""
Bringing the database up to date with the FPL API.

Adds fixtures, results and players that have appeared since the last update,
refreshes player attributes, and records transactions for players the entry has
bought or sold.
"""

from typing import Any

from sqlalchemy.orm.session import Session

from airsenal.core.caching import clear_query_caches
from airsenal.core.console import console
from airsenal.core.logging import get_logger
from airsenal.db.models import Player
from airsenal.db.queries.gameweeks import (
    get_last_complete_gameweek_in_db,
    next_gameweek,
)
from airsenal.db.queries.players import list_players
from airsenal.db.queries.teams import database_is_empty
from airsenal.db.queries.transactions import count_transactions
from airsenal.db.session import session_scope
from airsenal.ingest.fixtures import fill_fixtures_from_api
from airsenal.ingest.player_attributes import fill_attributes_table_from_api
from airsenal.ingest.player_mappings import add_mappings
from airsenal.ingest.player_scores import fill_playerscores_from_api
from airsenal.ingest.players import find_player_in_table
from airsenal.ingest.results import fill_results_from_api
from airsenal.remote.fpl_api import get_fetcher
from airsenal.squad.history import update_squad

logger = get_logger(__name__)


def update_transactions(season: str, fpl_team_id: int, dbsession: Session) -> bool:
    """Bring the transactions table up to date with the FPL API."""
    if next_gameweek(fetcher=get_fetcher()) != 1:
        logger.info("Checking team")
        n_transfers_api = len(get_fetcher().get_fpl_transfer_data(fpl_team_id))
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
    Fill in the gameweeks that have finished since the database was last updated.

    Updates the results, playerscore and, optionally, attributes tables. A no-op
    when the database is already level with the last finished gameweek.
    """
    last_in_db = get_last_complete_gameweek_in_db(season, dbsession=dbsession)
    if not last_in_db:
        # no results in database for this season yet
        last_in_db = 0
    last_finished = get_fetcher().get_last_finished_gameweek()

    if next_gameweek(fetcher=get_fetcher()) == 1:
        logger.info("Skipping team and result updates - season hasn't started.")
    elif last_finished > last_in_db:
        # need to update
        logger.info("Updating results table ...")
        fill_results_from_api(
            gw_start=last_in_db + 1,
            gw_end=next_gameweek(fetcher=get_fetcher()),
            season=season,
            dbsession=dbsession,
        )
        logger.info("Updating playerscores table ...")
        fill_playerscores_from_api(
            season=season,
            gw_start=last_in_db + 1,
            gw_end=next_gameweek(fetcher=get_fetcher()),
            dbsession=dbsession,
        )
    else:
        logger.info("Matches and player-scores already up-to-date")
    return True


def update_players(season: str, dbsession: Session) -> int:
    """Add any players the FPL API has that the player table does not."""
    players_from_db = list_players(
        position="all", team="all", season=season, dbsession=dbsession
    )
    player_data_from_api = get_fetcher().get_player_summary_data()
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
    players_from_db: list[Player],
    players_from_api: list[int],
    player_data_from_api: dict[int, dict[str, Any]],
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
    """
    Refresh player attributes from the last complete gameweek onwards.

    That gameweek is included rather than skipped: prices and availability can
    change after its matches finish but before the next deadline.
    """
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
        # fixtures may have moved between gameweeks, so any cached gameweek
        # a lookup from earlier in this process would be stale
        clear_query_caches()
        # update results and playerscores
        update_results(season, session)
        # update our squad
        update_transactions(season, fpl_team_id, session)
    return True


def update_database(season: str, attributes: bool, fpl_team_id: int | None) -> None:
    """Update database tables from current FPL data."""
    fpl_team_id = fpl_team_id or get_fetcher().FPL_TEAM_ID
    if not fpl_team_id:
        msg = "FPL team ID must be specified in args, config, or env"
        raise ValueError(msg)

    with session_scope() as session:
        if database_is_empty(session):
            logger.warning("Database is empty, run 'airsenal db create' first")
            return

        update_db(season, attributes, fpl_team_id, session)
