"""
Find multiple players with the same name in the same season using player summary data.
"""

import pandas as pd

from airsenal.framework.season import CURRENT_SEASON
from airsenal.scripts.make_player_summary import SAVE_FILE as SUMMARY_FILE


def find_duplicate_names(season: str = CURRENT_SEASON) -> None:
    df = pd.read_json(SUMMARY_FILE.format(season))
    name_groups = df.groupby("name")
    name_counts = name_groups["opta_code"].nunique()
    dup = name_counts > 1

    if dup.sum() > 0:
        print("\nDuplicated player names (and their Opta IDs):\n")
        codes = name_groups["opta_code"].unique()
        print(codes[dup])
    else:
        print(f"No duplicated player names found in {season} season.")


if __name__ == "__main__":
    find_duplicate_names()
