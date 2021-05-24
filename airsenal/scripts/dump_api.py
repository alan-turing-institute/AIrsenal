import json
from airsenal.framework.season import CURRENT_SEASON
from airsenal.framework.utils import fetcher


sdata = fetcher.get_current_summary_data()
with open(f"../data/FPL_{CURRENT_SEASON}.json", "w") as f:
    json.dump(sdata, f)

fixtures = fetcher.get_fixture_data()
with open(f"../data/fixture_data_{CURRENT_SEASON}.json", "w") as f:
    json.dump(fixtures, f)
