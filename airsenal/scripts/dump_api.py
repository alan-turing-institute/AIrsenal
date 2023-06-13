import json
import os

from airsenal.framework.season import CURRENT_SEASON
from airsenal.framework.utils import fetcher
from airsenal.scripts.make_player_details import make_player_details
from airsenal.scripts.make_player_summary import make_player_summary
from airsenal.scripts.make_results import make_results
from airsenal.scripts.scrape_transfermarkt import scrape_transfermarkt


def main():
    """
    Save all data from the FPL API and other sources, e.g. at the end of the season.
    """
    REPO_HOME = os.path.join(os.path.dirname(__file__), "..", "data")

    print("Saving summary data...")
    sdata = fetcher.get_current_summary_data()
    with open(os.path.join(REPO_HOME, f"FPL_{CURRENT_SEASON}.json"), "w") as f:
        json.dump(sdata, f)

    print("Saving fixture data...")
    fixtures = fetcher.get_fixture_data()
    with open(os.path.join(REPO_HOME, f"fixture_data_{CURRENT_SEASON}.json"), "w") as f:
        json.dump(fixtures, f)

    print("Saving team history data...")
    history = fetcher.get_fpl_team_history_data()
    with open(
        os.path.join(REPO_HOME, f"airsenal_history_{CURRENT_SEASON}.json"), "w"
    ) as f:
        json.dump(history, f)

    print("Saving transfer data...")
    transfers = fetcher.get_fpl_transfer_data()
    with open(
        os.path.join(REPO_HOME, f"airsenal_transfer_{CURRENT_SEASON}.json"), "w"
    ) as f:
        json.dump(transfers, f)

    print("Saving team data...")
    gws = [fetcher.get_fpl_team_data(gw) for gw in range(1, 39)]
    with open(os.path.join(REPO_HOME, f"airsenal_gw_{CURRENT_SEASON}.json"), "w") as f:
        json.dump(gws, f)

    print("Making player summary data file...")
    make_player_summary(CURRENT_SEASON)

    print("Making player details data file...")
    make_player_details([CURRENT_SEASON])

    print("Making results file...")
    make_results(CURRENT_SEASON)

    print("Scraping Transfermarkt data...")
    scrape_transfermarkt([CURRENT_SEASON])

    print("DONE!")


if __name__ == "__main__":
    main()
