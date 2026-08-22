"""Writing the FPL API's own expected points onto player attributes."""

import os
import uuid

from sqlalchemy.orm import Session

from airsenal.core.logging import get_logger
from airsenal.db.models import PlayerPrediction
from airsenal.db.queries.gameweeks import next_gameweek
from airsenal.db.queries.players import get_player_from_api_id
from airsenal.db.session import get_session
from airsenal.fetch.fpl_api import get_fetcher

logger = get_logger(__name__)


def fill_ep(csv_filename: str, dbsession: Session | None = None) -> None:
    """
    Fetch predicted points from the API and write to CSV and database.
    """
    dbsession = dbsession if dbsession is not None else get_session()
    if not os.path.exists(csv_filename):
        with open(csv_filename, "w") as outfile:
            outfile.write("player_id,gameweek,EP\n")

    tag = f"EP-{uuid.uuid4()!s}"
    summary_data = get_fetcher().get_player_summary_data()
    gameweek = next_gameweek()

    with open(csv_filename, "a") as outfile:
        for k, v in summary_data.items():
            player = get_player_from_api_id(k)
            if player is None:
                logger.warning("Player with API ID %s not found in database", k)
                continue

            player_id = player.player_id
            outfile.write(f"{player_id},{gameweek},{v['ep_next']}\n")

            pp = PlayerPrediction()
            pp.player_id = player_id
            pp.fixture.gameweek = gameweek
            pp.predicted_points = v["ep_next"]
            pp.tag = tag
            dbsession.add(pp)

    dbsession.commit()
