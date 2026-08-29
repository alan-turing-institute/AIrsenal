"""Dumping the database contents to CSV, one file per table."""

import csv
from pathlib import Path
from typing import TextIO

from sqlalchemy import select

from airsenal.core.data_files import data_file
from airsenal.core.logging import get_logger
from airsenal.db.models import (
    Base,
    FifaTeamRating,
    Fixture,
    Player,
    PlayerAttributes,
    PlayerScore,
    Result,
    Team,
    Transaction,
)
from airsenal.db.session import get_session

logger = get_logger(__name__)


def dump_db() -> None:
    """Write every table out to its own CSV in the packaged data directory."""
    # Dump Player database
    player_fieldnames = ["player_id", "fpl_api_id", "name", "opta_code"]
    save_table_fields(
        "players.csv",
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
        "chance_of_playing_next_round",
        "news",
        "return_gameweek",
        "price",
        "team",
        "position",
        "transfers_balance",
        "selected",
        "transfers_in",
        "transfers_out",
    ]
    save_table_fields(
        "player_attributes.csv",
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
        "fixtures.csv",
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
        "results.csv",
        result_fieldnames,
        Result,
        " ==== dumped Result database === ",
    )

    # Dump Team database
    team_fieldnames = ["id", "name", "full_name", "season", "team_id"]
    save_table_fields(
        "teams.csv",
        team_fieldnames,
        Team,
        " ==== dumped Team database === ",
    )

    # Dump FifaTeamRating database
    # Add season to the fieldnames once the table creation is updated
    fifa_team_rating_fieldnames = ["id", "season", "team", "att", "defn", "mid", "ovr"]
    save_table_fields(
        "fifa_team_ratings.csv",
        fifa_team_rating_fieldnames,
        FifaTeamRating,
        " ==== dumped FifaTeamRating database === ",
    )

    # Dump Transaction database
    transaction_fieldnames = [
        "id",
        "fpl_team_id",
        "free_hit",
        "time",
        "player_id",
        "gameweek",
        "bought_or_sold",
        "season",
        "tag",
        "price",
    ]
    save_table_fields(
        "transactions.csv",
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
        "expected_assists",
        "expected_goals",
        "expected_goal_involvements",
        "expected_goals_conceded",
        "clearances_blocks_interceptions",
        "defensive_contribution",
        "recoveries",
        "tackles",
    ]
    save_table_fields(
        "player_scores.csv",
        player_score_fieldnames,
        PlayerScore,
        " ==== dumped PlayerScore database === ",
    )


def save_table_fields(
    filename: str, fields: list[str], dbclass: type[Base], msg: str
) -> Path:
    result = data_file(filename)
    with result.open("w") as csvfile:
        write_rows_to_csv(csvfile, fields, dbclass)
    logger.info(msg)

    return result


def write_rows_to_csv(
    csvfile: TextIO, fieldnames: list[str], dbclass: type[Base]
) -> None:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    logger.info("Writing table %s", dbclass)
    for player in get_session().scalars(select(dbclass)).all():
        player_dict = vars(player)
        row = {
            field: value
            for field, value in player_dict.items()
            if isinstance(value, str | int | float)
        }

        writer.writerow(row)
