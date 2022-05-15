import json

from airsenal.framework.season import CURRENT_SEASON
from airsenal.framework.utils import fetcher


def main():
    """Save all data from the FPL API, e.g. at the end of the season."""
    sdata = fetcher.get_current_summary_data()
    with open(f"../data/FPL_{CURRENT_SEASON}.json", "w") as f:
        json.dump(sdata, f)

    fixtures = fetcher.get_fixture_data()
    with open(f"../data/fixture_data_{CURRENT_SEASON}.json", "w") as f:
        json.dump(fixtures, f)

    history = fetcher.get_fpl_team_history_data()
    with open(f"../data/airsenal_history_{CURRENT_SEASON}.json", "w") as f:
        json.dump(history, f)

    transfers = fetcher.get_fpl_transfer_data()
    with open(f"../data/airsenal_transfer_{CURRENT_SEASON}.json", "w") as f:
        json.dump(transfers, f)

    gws = [fetcher.get_fpl_team_data(gw) for gw in range(1, 39)]
    with open(f"../data/airsenal_gw_{CURRENT_SEASON}.json", "w") as f:
        json.dump(gws, f)


if __name__ == "__main__":
    main()
