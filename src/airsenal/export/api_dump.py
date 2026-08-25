import json
import os

from airsenal.core.data_files import data_dir
from airsenal.core.logging import get_logger
from airsenal.export.player_details import make_player_details
from airsenal.export.player_summary import make_player_summary
from airsenal.export.results import make_results
from airsenal.game.season import CURRENT_SEASON
from airsenal.remote.fpl_api import get_fetcher
from airsenal.remote.transfermarkt import scrape_transfermarkt

logger = get_logger(__name__)


def main() -> None:
    """
    Save all data from the FPL API and other sources, e.g. at the end of the season.
    """
    repo_home = data_dir()

    logger.info("Saving summary data...")
    sdata = get_fetcher().get_current_summary_data()
    with open(os.path.join(repo_home, f"FPL_{CURRENT_SEASON}.json"), "w") as f:
        json.dump(sdata, f)

    logger.info("Saving fixture data...")
    fixtures = get_fetcher().get_fixture_data()
    with open(os.path.join(repo_home, f"fixture_data_{CURRENT_SEASON}.json"), "w") as f:
        json.dump(fixtures, f)

    logger.info("Saving team history data...")
    history = get_fetcher().get_fpl_team_history_data()
    with open(
        os.path.join(repo_home, f"airsenal_history_{CURRENT_SEASON}.json"), "w"
    ) as f:
        json.dump(history, f)

    logger.info("Saving transfer data...")
    transfers = get_fetcher().get_fpl_transfer_data()
    with open(
        os.path.join(repo_home, f"airsenal_transfer_{CURRENT_SEASON}.json"), "w"
    ) as f:
        json.dump(transfers, f)

    logger.info("Saving team data...")
    gws = [get_fetcher().get_fpl_team_data(gw) for gw in range(1, 39)]
    with open(os.path.join(repo_home, f"airsenal_gw_{CURRENT_SEASON}.json"), "w") as f:
        json.dump(gws, f)

    logger.info("Making player summary data file...")
    make_player_summary(CURRENT_SEASON)

    logger.info("Making player details data file...")
    make_player_details(CURRENT_SEASON)

    logger.info("Making results file...")
    make_results(CURRENT_SEASON)

    logger.info("Scraping Transfermarkt data...")
    scrape_transfermarkt([CURRENT_SEASON])

    logger.info("DONE!")
