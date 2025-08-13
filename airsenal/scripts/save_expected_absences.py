"""
Week by week, we will run this script to save the information we get from the FPL API on
players that have a <100% chance of playing the next gameweek.
This will make it easier in the future to replay the season.

The data will be written in the same way as the existing `absences_yyyy.csv`
files, where up until the 24/25 season, these files were retrospectively created
by scraping external websites.   From 25/26 onwards, the data will be in the
same format, but will be the actual FPL API data.
"""

import datetime
import os

from airsenal.framework.schema import Absence, PlayerAttributes
from airsenal.framework.utils import CURRENT_SEASON, session


def save_absences(absence_list):
    """
    Write the latest data to the "absences_yyyy.csv" file
    """
    columns = [column.key for column in Absence.__table__.columns]
    columns.remove("id")
    # add player name as the first column
    columns = ["player", *columns]
    # first column
    REPO_HOME = os.path.join(os.path.dirname(__file__), "..", "data")
    output_file = os.path.join(REPO_HOME, f"absences_{CURRENT_SEASON}.csv")
    # create file if it doesn't exist yet, and write column headers
    if not os.path.exists(output_file):
        with open(output_file, "w") as outfile:
            outfile.write(",".join(columns) + "\n")
    # Write the new data

    with open(output_file, "a") as outfile:
        for a in absence_list:
            row = ",".join([str(a.__getattribute__(c)) for c in columns])
            outfile.write(row + "\n")


def player_attribute_to_absence(player_attribute):
    """
    Convert a PlayerAttribute row, which has data from the FPL API on player
    unavailability, into an Absence row, which has the columns that we will
    write to csv.

    Parameters
    ==========
    player_attribute: PlayerAttribute row

    Returns
    =======
    a: Absence row, with relevant details copied across
    """
    a = Absence()
    a.player = player_attribute.player
    a.player_id = player_attribute.player_id
    a.season = player_attribute.season
    a.gw_from = player_attribute.gameweek
    a.gw_until = player_attribute.return_gameweek
    a.chance_of_playing = player_attribute.chance_of_playing_next_round
    a.reason = player_attribute.news
    a.timestamp = datetime.datetime.now().isoformat()

    return a


def main():
    """
    main function, to be used as entrypoint.
    """
    pas = (
        session.query(PlayerAttributes)
        .filter(PlayerAttributes.season == CURRENT_SEASON)
        .filter(PlayerAttributes.chance_of_playing_next_round != None)  # noqa: E711
        .all()
    )
    print(f"Found {len(pas)} player absences.")
    absences = [player_attribute_to_absence(pa) for pa in pas]
    save_absences(absences)
