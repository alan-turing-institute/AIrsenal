"""
Script to dump the database contents.
"""

import csv
import os

from airsenal.framework.schema import (
    FifaTeamRating,
    Fixture,
    Player,
    PlayerAttributes,
    PlayerScore,
    Result,
    Team,
    Transaction,
)
from airsenal.framework.utils import session


def main():
    # Dump Player database
    player_fieldnames = ["player_id", "name"]
    save_table_fields(
        "../data/players.csv",
        player_fieldnames,
        Player,
        " ==== dumped Player database === ",
    )

    # Dump PlayerAttributes database
    player_attributes_fieldnames = [
        "id",
        "player_id",
        "season",
        "gameweek",
        "price",
        "team",
        "position",
        "transfers_balance",
        "selected",
        "transfers_in",
        "transfers_out",
    ]
    save_table_fields(
        "../data/player_attributes.csv",
        player_attributes_fieldnames,
        PlayerAttributes,
        " ==== dumped PlayerAttributes database === ",
    )

    # Dump Fixture database
    fixture_fieldnames = [
        "fixture_id",
        "date",
        "gameweek",
        "home_team",
        "away_team",
        "season",
        "tag",
        "player_id",
    ]
    save_table_fields(
        "../data/fixtures.csv",
        fixture_fieldnames,
        Fixture,
        " ==== dumped Fixture database === ",
    )

    # Dump Result database
    result_fieldnames = [
        "result_id",
        "fixture_id",
        "home_score",
        "away_score",
        "player_id",
    ]
    save_table_fields(
        "../data/results.csv",
        result_fieldnames,
        Result,
        " ==== dumped Result database === ",
    )

    # Dump Team database
    team_fieldnames = ["id", "name", "full_name", "season", "team_id"]
    save_table_fields(
        "../data/teams.csv",
        team_fieldnames,
        Team,
        " ==== dumped Team database === ",
    )

    # Dump FifaTeamRating database
    # Add season to the fieldnames once the table creation is updated
    fifa_team_rating_fieldnames = ["team", "att", "defn", "mid", "ovr"]
    save_table_fields(
        "../data/fifa_team_ratings.csv",
        fifa_team_rating_fieldnames,
        FifaTeamRating,
        " ==== dumped FifaTeamRating database === ",
    )

    # Dump Transaction database
    transaction_fieldnames = [
        "id",
        "player_id",
        "gameweek",
        "bought_or_sold",
        "season",
        "tag",
        "price",
    ]
    save_table_fields(
        "../data/transactions.csv",
        transaction_fieldnames,
        Transaction,
        " ==== dumped Transaction database === ",
    )

    # Dump PlayerScore database
    player_score_fieldnames = [
        "id",
        "player_team",
        "opponent",
        "points",
        "goals",
        "assists",
        "bonus",
        "conceded",
        "minutes",
        "player_id",
        "result_id",
        "fixture_id",
        "clean_sheets",
        "own_goals",
        "penalties_saved",
        "penalties_missed",
        "yellow_cards",
        "red_cards",
        "saves",
        "bps",
        "influence",
        "creativity",
        "threat",
        "ict_index",
        "value",
        "transfers_balance",
        "selected",
        "transfers_in",
        "transfers_out",
    ]
    save_table_fields(
        "../data/player_scores.csv",
        player_score_fieldnames,
        PlayerScore,
        " ==== dumped PlayerScore database === ",
    )


def save_table_fields(path, fields, dbclass, msg):
    result = os.path.join(os.path.dirname(__file__), path)
    with open(result, "w") as csvfile:
        write_rows_to_csv(csvfile, fields, dbclass)
    print(msg)

    return result


def write_rows_to_csv(csvfile, fieldnames, dbclass):
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for player in session.query(dbclass).all():
        player = vars(player)
        row = {
            field: player[field]
            for field, value____ in player.items()
            if isinstance(value____, (str, int, float))
        }

        writer.writerow(row)


if __name__ == "__main__":
    print(" ==== dumping database contents === ")
    main()
