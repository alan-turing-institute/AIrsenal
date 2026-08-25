import pandas as pd

from airsenal.core.data_files import data_file
from airsenal.core.logging import get_logger
from airsenal.game.season import CURRENT_SEASON
from airsenal.remote.fpl_api import get_fetcher

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
    teams.to_csv(data_file(f"teams_{CURRENT_SEASON}.csv"), index=False)

    logger.info("%s", teams)
    logger.info("DONE!")


if __name__ == "__main__":
    main()
