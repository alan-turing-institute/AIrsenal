"""Script to fill the database after install."""

from sqlalchemy.orm.session import Session

from airsenal.core.console import console
from airsenal.core.logging import get_logger
from airsenal.core.season import CURRENT_SEASON, get_past_seasons, sort_seasons
from airsenal.db.queries.teams import database_is_empty
from airsenal.db.session import clean_database, session_scope
from airsenal.fetch.fpl_api import get_fetcher
from airsenal.ingest.absences import make_absence_table
from airsenal.ingest.fifa_ratings import make_fifa_ratings_table
from airsenal.ingest.fixtures import make_fixture_table
from airsenal.ingest.player_attributes import make_attributes_table
from airsenal.ingest.player_scores import make_playerscore_table
from airsenal.ingest.players import make_player_table
from airsenal.ingest.results import make_result_table
from airsenal.ingest.teams import make_team_table
from airsenal.squad.history import fill_initial_squad

logger = get_logger(__name__)


def check_clean_db(clean: bool, dbsession: Session) -> bool:
    """Check whether an AIrsenal database already exists. If clean is True attempt to
    delete any pre-existing database first. Returns True if database exists and is not
    empty.
    """
    if clean:
        logger.info("Cleaning database...")
        clean_database()
    return database_is_empty(dbsession)


def make_init_db(
    fpl_team_id: int | None, seasons: list[str], dbsession: Session
) -> bool:
    with console.status("Creating the database..."):
        seasons = sort_seasons(seasons)
        make_team_table(seasons=seasons, dbsession=dbsession)
        make_fixture_table(seasons=seasons, dbsession=dbsession)
        make_result_table(seasons=seasons, dbsession=dbsession)
        make_fifa_ratings_table(seasons=seasons, dbsession=dbsession)

        make_player_table(seasons=seasons, dbsession=dbsession)
        make_attributes_table(seasons=seasons, dbsession=dbsession)
        make_playerscore_table(seasons=seasons, dbsession=dbsession)
        make_absence_table(seasons=seasons, dbsession=dbsession)

        if CURRENT_SEASON in seasons:
            if fpl_team_id is None:
                msg = "FPL team ID must be specified in args, config, or env"
                raise ValueError(msg)
            fill_initial_squad(fpl_team_id=fpl_team_id, dbsession=dbsession)

    logger.info("DONE!")
    return not database_is_empty(dbsession)


def create_database(
    fpl_team_id: int | None,
    clean: bool,
    n_previous: int,
    include_current_season: bool = True,
) -> None:
    """Create the database, including historical and current-season data."""
    if include_current_season:
        fpl_team_id = fpl_team_id or get_fetcher().FPL_TEAM_ID
    with session_scope() as dbsession:
        continue_setup = check_clean_db(clean, dbsession)
        if continue_setup:
            past = get_past_seasons(n_previous)
            seasons = [CURRENT_SEASON, *past] if include_current_season else past
            make_init_db(fpl_team_id, seasons, dbsession)
        else:
            logger.info(
                "AIrsenal database already exists. "
                "Run 'airsenal db create --clean' to delete and recreate it,\n"
                "or keep the current database and continue to 'airsenal db update'."
            )
