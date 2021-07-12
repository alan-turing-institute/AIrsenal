"""
Script to dump the database contents.
"""

import csv
import os

from airsenal.framework.schema import (
    Player,
    PlayerAttributes,
    Fixture,
    Result,
    Team,
    FifaTeamRating,
    Transaction,
    PlayerScore,
)
from airsenal.framework.utils import session


def main():

    # Dump Player database
    player_fieldnames = ["player_id", "name"]
    output_path = os.path.join(os.path.dirname(__file__), "../data/players.csv")
    with open(output_path, "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=player_fieldnames)
        writer.writeheader()
        for player in session.query(Player).all():
            player = vars(player)
            row = {
                field: player[field]
                for field, value____ in player.items()
                if isinstance(value____, (str, int, float))
            }

            writer.writerow(row)
    print(" ==== dumped Player database === ")

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
    output_path = os.path.join(
        os.path.dirname(__file__), "../data/player_attributes.csv"
    )
    with open(output_path, "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=player_attributes_fieldnames)
        writer.writeheader()
        for player_attributes in session.query(PlayerAttributes).all():
            player_attributes = vars(player_attributes)
            row = {
                field: player_attributes[field]
                for field, value_ in player_attributes.items()
                if isinstance(value_, (str, int, float))
            }

            writer.writerow(row)
    print(" ==== dumped PlayerAttributes database === ")

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
    output_path = os.path.join(os.path.dirname(__file__), "../data/fixtures.csv")
    with open(output_path, "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fixture_fieldnames)
        writer.writeheader()
        for fixture in session.query(Fixture).all():
            fixture = vars(fixture)
            row = {
                field: fixture[field]
                for field, value__ in fixture.items()
                if isinstance(value__, (str, int, float))
            }

            writer.writerow(row)
    print(" ==== dumped Fixture database === ")

    # Dump Result database
    result_fieldnames = [
        "result_id",
        "fixture_id",
        "home_score",
        "away_score",
        "player_id",
    ]
    output_path = os.path.join(os.path.dirname(__file__), "../data/results.csv")
    with open(output_path, "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=result_fieldnames)
        writer.writeheader()
        for result in session.query(Result).all():
            result = vars(result)
            row = {
                field: result[field]
                for field, value_____ in result.items()
                if isinstance(value_____, (str, int, float))
            }

            writer.writerow(row)
    print(" ==== dumped Result database === ")

    # Dump Team database
    team_fieldnames = ["id", "name", "full_name", "season", "team_id"]
    output_path = os.path.join(os.path.dirname(__file__), "../data/teams.csv")
    with open(output_path, "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=team_fieldnames)
        writer.writeheader()
        for team in session.query(Team).all():
            team = vars(team)
            row = {
                field: team[field]
                for field, value___ in team.items()
                if isinstance(value___, (str, int, float))
            }

            writer.writerow(row)
    print(" ==== dumped Team database === ")

    # Dump FifaTeamRating database
    # Add season to the fieldnames once the table creation is updated
    fifa_team_rating_fieldnames = ["team", "att", "defn", "mid", "ovr"]
    output_path = os.path.join(
        os.path.dirname(__file__), "../data/fifa_team_ratings.csv"
    )
    with open(output_path, "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fifa_team_rating_fieldnames)
        writer.writeheader()
        for fifa_team_rating in session.query(FifaTeamRating).all():
            fifa_team_rating = vars(fifa_team_rating)
            row = {}
            for field, value in fifa_team_rating.items():
                if isinstance(value, (str, int, float)):
                    row[field] = fifa_team_rating[field]
            writer.writerow(row)
    print(" ==== dumped FifaTeamRating database === ")

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
    output_path = os.path.join(os.path.dirname(__file__), "../data/transactions.csv")
    with open(output_path, "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=transaction_fieldnames)
        writer.writeheader()
        for transaction in session.query(Transaction).all():
            transaction = vars(transaction)
            row = {}
            for field, value______ in transaction.items():
                if isinstance(value______, (str, int, float)):
                    row[field] = transaction[field]
            writer.writerow(row)
    print(" ==== dumped Transaction database === ")

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
    output_path = os.path.join(os.path.dirname(__file__), "../data/player_scores.csv")
    with open(output_path, "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=player_score_fieldnames)
        writer.writeheader()
        for player_score in session.query(PlayerScore).all():
            player_score = vars(player_score)
            row = {}
            for field, value_______ in player_score.items():
                if isinstance(value_______, (str, int, float)):
                    row[field] = player_score[field]
            writer.writerow(row)
    print(" ==== dumped PlayerScore database === ")


if __name__ == "__main__":
    print(" ==== dumping database contents === ")
    main()
