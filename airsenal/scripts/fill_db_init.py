"""Script to fill the database after install."""

from sqlalchemy.orm.session import Session

from airsenal.framework.output import get_logger
from airsenal.framework.schema import clean_database, database_is_empty, session_scope
from airsenal.framework.season import CURRENT_SEASON, sort_seasons
from airsenal.framework.transaction_utils import fill_initial_squad
from airsenal.framework.utils import fetcher, get_past_seasons
from airsenal.scripts.fill_absence_table import make_absence_table
from airsenal.scripts.fill_fifa_ratings_table import make_fifa_ratings_table
from airsenal.scripts.fill_fixture_table import make_fixture_table
from airsenal.scripts.fill_player_attributes_table import make_attributes_table
from airsenal.scripts.fill_player_table import make_player_table
from airsenal.scripts.fill_playerscore_table import make_playerscore_table
from airsenal.scripts.fill_result_table import make_result_table
from airsenal.scripts.fill_team_table import make_team_table

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
    no_current_season: bool,
) -> None:
    """Create the database, including historical and current-season data."""
    if not no_current_season:
        fpl_team_id = fpl_team_id or fetcher.FPL_TEAM_ID
    with session_scope() as dbsession:
        continue_setup = check_clean_db(clean, dbsession)
        if continue_setup:
            if no_current_season:
                seasons = get_past_seasons(n_previous)
            else:
                seasons = [CURRENT_SEASON, *get_past_seasons(n_previous)]
            make_init_db(fpl_team_id, seasons, dbsession)
        else:
            logger.info(
                "AIrsenal database already exists. "
                "Run 'airsenal db create --clean' to delete and recreate it,\n"
                "or keep the current database and continue to 'airsenal db update'."
            )
