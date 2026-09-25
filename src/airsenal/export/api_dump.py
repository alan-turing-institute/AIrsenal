"""
Saving everything AIrsenal fetches to the packaged data files.

Writes the season's player details, summaries, results and Transfermarkt data into
`src/airsenal/data/`.
"""

import json
from typing import Any

from airsenal.core.data_files import data_dir
from airsenal.core.logging import get_logger
from airsenal.export.player_details import make_player_details
from airsenal.export.player_summary import make_player_summary
from airsenal.export.results import make_results
from airsenal.game.season import CURRENT_SEASON
from airsenal.remote.fpl_api import get_fetcher
from airsenal.remote.transfermarkt import scrape_transfermarkt

logger = get_logger(__name__)


def _dump(data: Any, filename: str) -> None:
    with (data_dir() / filename).open("w") as f:
        json.dump(data, f)


def dump_api() -> None:
    """Save everything from the FPL API and other sources."""
    logger.info("Saving summary data...")
    _dump(get_fetcher().get_current_summary_data(), f"FPL_{CURRENT_SEASON}.json")

    logger.info("Saving fixture data...")
    _dump(get_fetcher().get_fixture_data(), f"fixture_data_{CURRENT_SEASON}.json")

    logger.info("Saving team history data...")
    _dump(
        get_fetcher().get_fpl_team_history_data(),
        f"airsenal_history_{CURRENT_SEASON}.json",
    )

    logger.info("Saving transfer data...")
    _dump(
        get_fetcher().get_fpl_transfer_data(),
        f"airsenal_transfer_{CURRENT_SEASON}.json",
    )

    logger.info("Saving team data...")
    _dump(
        [get_fetcher().get_fpl_team_data(gameweek) for gameweek in range(1, 39)],
        f"airsenal_gw_{CURRENT_SEASON}.json",
    )

    logger.info("Making player summary data file...")
    make_player_summary(CURRENT_SEASON)

    logger.info("Making player details data file...")
    make_player_details(CURRENT_SEASON)

    logger.info("Making results file...")
    make_results(CURRENT_SEASON)

    logger.info("Scraping Transfermarkt data...")
    scrape_transfermarkt([CURRENT_SEASON])

    logger.info("DONE!")
