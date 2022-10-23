"""
Find multiple players with the same name in the same season from a locally cloned copy
of the https://github.com/vaastav/Fantasy-Premier-League repository on GitHub.
"""
from glob import glob
from typing import List, Union

import pandas as pd

from airsenal.framework.utils import get_past_seasons
from airsenal.scripts.make_player_details import (
    PLAYERS_DIR,
    get_long_season_name,
    path_to_index,
    path_to_name,
)


def find_duplicate_names(seasons: Union[str, List[str]] = get_past_seasons(6)) -> None:
    if isinstance(seasons, str):
        seasons = [seasons]

    output = []

    for season in seasons:
        season_longname = get_long_season_name(season)
        print(f"Doing {season}...")

        # names of all player directories for this season
        sub_dirs = glob(PLAYERS_DIR.format(season_longname) + "/*/")

        for directory in sub_dirs:
            name = path_to_name(directory)
            idx = path_to_index(directory)

            player_dict = {
                "name": name,
                "season": season,
                "id": idx,
            }
            output.append(player_dict)

    print("\nDuplicated player names (and their IDs):\n")
    df = pd.DataFrame(output)
    nunique = df.groupby(["season", "name"])["id"].nunique()
    dup = nunique > 1
    ids = df.groupby(["season", "name"])["id"].unique()
    print(ids[dup])


if __name__ == "__main__":
    find_duplicate_names()
