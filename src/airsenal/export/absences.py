"""
Saving the players the FPL API says have a <100% chance of playing next gameweek.

Run week by week (`airsenal run --save-absences`), so that a season can be
replayed later. The data is written in the same format as the packaged
`absences_yyyy.csv` files, which are retrospectively scraped from Transfermarkt
by `remote.transfermarkt`, so that `ingest.absences.load_absences` can read
either.
"""

import csv
import os
from datetime import date, datetime

from sqlalchemy import select
from sqlalchemy.orm.session import Session

from airsenal.core.data_files import FilePath, absences_file
from airsenal.core.logging import get_logger
from airsenal.db.models import Fixture, PlayerAttributes
from airsenal.db.session import get_session
from airsenal.game.season import CURRENT_SEASON

logger = get_logger(__name__)

# The header of the absences_yyyy.csv files, in the order load_absences expects.
# "days" and "games" are carried over from the Transfermarkt-scraped files; they are
# written for consistency but load_absences does not read them.
ABSENCE_CSV_COLUMNS = (
    "season",
    "details",
    "from",
    "until",
    "days",
    "games",
    "reason",
    "player",
    "url",
)

SUSPENSION_KEYWORDS = ("suspend", "ban", "red card")
INJURY_KEYWORDS = ("injur", "knock", "strain", "problem", "surgery", "ill", "virus")


def classify_reason(news: str | None) -> str:
    """
    Map the FPL API's news text onto a reason for the absences_yyyy.csv files.

    Args:
        news: The API's `news` field, e.g. "Knee injury - Expected back 25 Dec".

    Returns:
        One of "injury", "suspension" or "absence".
    """
    if not news:
        return "absence"
    lowered = news.lower()
    if any(word in lowered for word in SUSPENSION_KEYWORDS):
        return "suspension"
    if any(word in lowered for word in INJURY_KEYWORDS):
        return "injury"
    return "absence"


def get_gameweek_start_date(
    gameweek: int, season: str, dbsession: Session
) -> date | None:
    """
    Date of the earliest fixture in a gameweek.

    Used as the start, or end, date of an absence that the FPL API only gives us
    in gameweeks.

    Returns:
        None if the gameweek has no scheduled fixtures.
    """
    dates = dbsession.scalars(
        select(Fixture.date).where(
            Fixture.season == season,
            Fixture.gameweek == gameweek,
            Fixture.date.is_not(None),
        )
    ).all()
    # The is_not(None) filter above is invisible to the type checker.
    parsed = [datetime.fromisoformat(d).date() for d in dates if d is not None]
    return min(parsed) if parsed else None


def read_existing_keys(path: FilePath) -> set[tuple[str, str]]:
    """
    The (player, from) pairs an absences csv file already holds.

    Repeated runs skip these rather than appending duplicate rows.
    """
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as infile:
        return {(row["player"], row["from"]) for row in csv.DictReader(infile)}


def player_attribute_to_row(
    player_attribute: PlayerAttributes, dbsession: Session
) -> dict[str, str] | None:
    """
    Convert a PlayerAttributes row into a row of the absences csv file.

    The attributes row is the FPL API's view of a player's unavailability.

    Returns:
        Csv column name to value, or None if the absence has no usable start date -
        `load_absences` skips such rows anyway.
    """
    season = player_attribute.season
    date_from = get_gameweek_start_date(player_attribute.gameweek, season, dbsession)
    if date_from is None:
        logger.warning(
            "No fixture dates for %s GW%s, skipping %s",
            season,
            player_attribute.gameweek,
            player_attribute.player,
        )
        return None

    return_gameweek = player_attribute.return_gameweek
    date_until = (
        get_gameweek_start_date(return_gameweek, season, dbsession)
        if return_gameweek is not None
        else None
    )

    days = (date_until - date_from).days if date_until is not None else ""
    games = (
        return_gameweek - player_attribute.gameweek
        if return_gameweek is not None
        else ""
    )
    news = player_attribute.news

    return {
        "season": season,
        # `details` is a str100 column in the Absence table, so keep it within that.
        "details": (news or "Unavailable")[:100],
        "from": date_from.isoformat(),
        "until": date_until.isoformat() if date_until is not None else "",
        "days": str(days),
        "games": str(games),
        "reason": classify_reason(news),
        "player": player_attribute.player.name,
        "url": "",  # the FPL API is the source; there is no per-player page to cite
    }


def save_absences(
    rows: list[dict[str, str]], season: str, path: FilePath | None = None
) -> int:
    """
    Append rows to the absences_yyyy.csv file.

    The file is created with a header if it is not there, and rows already
    present are skipped.

    Returns:
        The number of rows actually written.
    """
    if path is None:
        path = absences_file(season)
    existing = read_existing_keys(path)
    new_rows = [r for r in rows if (r["player"], r["from"]) not in existing]

    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=ABSENCE_CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(new_rows)

    logger.info(
        "Wrote %s new absences to %s (%s already present).",
        len(new_rows),
        path,
        len(rows) - len(new_rows),
    )
    return len(new_rows)


def save_expected_absences() -> None:
    """Write every current-season absence known to the database out to its CSV."""
    dbsession = get_session()
    attributes = dbsession.scalars(
        select(PlayerAttributes).where(
            PlayerAttributes.season == CURRENT_SEASON,
            PlayerAttributes.chance_of_playing_next_round.is_not(None),
            PlayerAttributes.chance_of_playing_next_round < 100,
        )
    ).all()
    logger.info("Found %s player absences.", len(attributes))
    rows = [player_attribute_to_row(pa, dbsession) for pa in attributes]
    save_absences([r for r in rows if r is not None], CURRENT_SEASON)
