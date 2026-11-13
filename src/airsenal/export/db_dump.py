"""Dumping the database contents to CSV, one file per table."""

import csv
from typing import TextIO

from sqlalchemy import inspect, select
from sqlalchemy.orm.session import Session

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

# The file each table is written to, in the packaged data directory.
DUMP_FILES: dict[str, type[Base]] = {
    "players.csv": Player,
    "player_attributes.csv": PlayerAttributes,
    "fixtures.csv": Fixture,
    "results.csv": Result,
    "teams.csv": Team,
    "fifa_team_ratings.csv": FifaTeamRating,
    "transactions.csv": Transaction,
    "player_scores.csv": PlayerScore,
}


def dump_db(dbsession: Session | None = None) -> None:
    """Write each table in `DUMP_FILES` out to its CSV, with one column per column."""
    for filename, dbclass in DUMP_FILES.items():
        save_table(filename, dbclass, dbsession=dbsession)


def save_table(
    filename: str, dbclass: type[Base], dbsession: Session | None = None
) -> None:
    with data_file(filename).open("w") as csvfile:
        write_rows_to_csv(csvfile, dbclass, dbsession=dbsession)
    logger.info(" ==== dumped %s database === ", dbclass.__name__)


def write_rows_to_csv(
    csvfile: TextIO, dbclass: type[Base], dbsession: Session | None = None
) -> None:
    """Write every row of a table, a null as an empty field."""
    fieldnames = [column.key for column in inspect(dbclass).column_attrs]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    logger.info("Writing table %s", dbclass)
    for record in get_session(dbsession).scalars(select(dbclass)).all():
        writer.writerow({field: getattr(record, field) for field in fieldnames})
