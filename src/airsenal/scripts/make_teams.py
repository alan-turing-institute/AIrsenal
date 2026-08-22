import os

import pandas as pd

from airsenal.core.logging import get_logger
from airsenal.domain.season import CURRENT_SEASON
from airsenal.fetch.fpl_api import get_fetcher

SCRIPT_DIR = os.path.dirname(__file__)

logger = get_logger(__name__)


def main():
    data = get_fetcher().get_current_summary_data()
    teams = pd.DataFrame(data["teams"])

    teams = teams[["short_name", "name", "id"]]
    teams.rename(
        columns={"short_name": "name", "name": "full_name", "id": "team_id"},
        inplace=True,
    )
    teams["season"] = CURRENT_SEASON

    teams = teams[["name", "full_name", "season", "team_id"]]
    teams.to_csv(f"{SCRIPT_DIR}/../data/teams_{CURRENT_SEASON}.csv", index=False)

    logger.info("%s", teams)
    logger.info("DONE!")


if __name__ == "__main__":
    main()
