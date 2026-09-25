"""Dumping the database contents to CSV, one file per table."""

import csv
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
    player_fieldnames = ["player_id", "fpl_api_id", "name", "opta_code"]
    save_table_fields("players.csv", player_fieldnames, Player)

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
        "player_attributes.csv", player_attributes_fieldnames, PlayerAttributes
    )

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
    save_table_fields("fixtures.csv", fixture_fieldnames, Fixture)

    result_fieldnames = [
        "result_id",
        "fixture_id",
        "home_score",
        "away_score",
        "player_id",
    ]
    save_table_fields("results.csv", result_fieldnames, Result)

    team_fieldnames = ["id", "name", "full_name", "season", "team_id"]
    save_table_fields("teams.csv", team_fieldnames, Team)

    fifa_team_rating_fieldnames = ["id", "season", "team", "att", "defn", "mid", "ovr"]
    save_table_fields(
        "fifa_team_ratings.csv", fifa_team_rating_fieldnames, FifaTeamRating
    )

    transaction_fieldnames = [
        "id",
        "fpl_team_id",
        "free_hit",
        "counts_as_transfer",
        "time",
        "player_id",
        "gameweek",
        "bought_or_sold",
        "season",
        "tag",
        "price",
    ]
    save_table_fields("transactions.csv", transaction_fieldnames, Transaction)

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
    save_table_fields("player_scores.csv", player_score_fieldnames, PlayerScore)


def save_table_fields(filename: str, fields: list[str], dbclass: type[Base]) -> None:
    with data_file(filename).open("w") as csvfile:
        write_rows_to_csv(csvfile, fields, dbclass)
    logger.info(" ==== dumped %s database === ", dbclass.__name__)


def write_rows_to_csv(
    csvfile: TextIO, fieldnames: list[str], dbclass: type[Base]
) -> None:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    logger.info("Writing table %s", dbclass)
    for record in get_session().scalars(select(dbclass)).all():
        row = {
            field: value
            for field, value in vars(record).items()
            if isinstance(value, str | int | float)
        }

        writer.writerow(row)
