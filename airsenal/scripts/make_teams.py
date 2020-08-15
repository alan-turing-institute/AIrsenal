import pandas as pd

from airsenal.framework.utils import CURRENT_SEASON, fetcher

data = fetcher.get_current_summary_data()
teams = pd.DataFrame(data["teams"])

teams = teams[["short_name", "name", "id"]]
teams.rename(
    columns={"short_name": "name", "name": "full_name", "id": "team_id"}, inplace=True
)
teams["season"] = CURRENT_SEASON

teams = teams[["name", "full_name", "season", "team_id"]]
teams.to_csv(f"../data/teams_{CURRENT_SEASON}.csv", index=False)

print(teams)
print("DONE!")
