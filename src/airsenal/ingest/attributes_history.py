"""
Reading the per-day player attributes history CSV.

`export/attributes.py` appends one row per player per day to
`player_attributes_history_{season}.csv` while a season is active, giving a record
of what the FPL API said about each player on each day. This is what reads it back,
for `ingest/player_attributes.py` (the status on a gameweek's first matchday) and
`ingest/player_scores.py` (the status on the morning of a particular kickoff).
"""

import datetime
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from airsenal.core.data_files import data_file
from airsenal.core.logging import get_logger
from airsenal.db.models import Player
from airsenal.game.season import CURRENT_SEASON, season_str_to_year
from airsenal.remote.download import download_with_resume
from airsenal.remote.errors import RemoteError

logger = get_logger(__name__)

# The first day the packaged per-day attributes history covers - the daily
# `airsenal dump attributes` job started appending on this date.
ATTRIBUTES_HISTORY_START = datetime.date(2025, 9, 12)

# The season that day falls in. There is no history at all before it.
ATTRIBUTES_HISTORY_FIRST_SEASON = "2526"

# Only these are read. The file is 23MB for a full season, and the columns left
# out (price, selected, the transfer counts) come from player_details instead.
ATTRIBUTES_HISTORY_COLUMNS = (
    "timestamp",
    "gameweek",
    "opta_code",
    "player",
    "news",
    "chance_of_playing_next_round",
    "return_gameweek",
)

# The packaged data lives at `src/airsenal/data` here and lived at
# `airsenal/data` before the move to a src layout. This downloads from `main`,
# which is on one side of that move or the other depending on whether the move
# has landed there yet, so try both: pinning either one alone means the download
# starts 404ing on the day the layout changes, and the caller turns that into a
# warning and carries on without the history.
_ATTRIBUTES_HISTORY_PATHS = ("src/airsenal/data", "airsenal/data")


@dataclass(frozen=True)
class Availability:
    """What the availability columns of one PlayerAttributes row should say."""

    news: str | None
    chance_of_playing_next_round: int | None
    return_gameweek: int | None


def has_attributes_history(season: str) -> bool:
    """Whether the daily attributes dump was running during a season."""
    return season_str_to_year(season) >= season_str_to_year(
        ATTRIBUTES_HISTORY_FIRST_SEASON
    )


def _attributes_history_urls(season: str) -> list[str]:
    """Where the attributes history for a season might be, best guess first."""
    return [
        "https://raw.githubusercontent.com/alan-turing-institute/AIrsenal/refs/"
        f"heads/main/{path}/player_attributes_history_{season}.csv"
        for path in _ATTRIBUTES_HISTORY_PATHS
    ]


def _read_history_csv(path: Path | str) -> pd.DataFrame:
    df_attributes = pd.read_csv(path, usecols=list(ATTRIBUTES_HISTORY_COLUMNS))
    df_attributes["day"] = pd.to_datetime(df_attributes["timestamp"]).dt.date
    return df_attributes


def _load_packaged(season: str) -> pd.DataFrame | None:
    path = data_file(f"player_attributes_history_{season}.csv")
    if not path.is_file():
        logger.info("No packaged attributes history at %s", path)
        return None
    return _read_history_csv(path)


def _load_downloaded(season: str) -> pd.DataFrame | None:
    for url in _attributes_history_urls(season):
        try:
            with tempfile.TemporaryDirectory(prefix="airsenal_attrs_") as tmpdir:
                tmp_csv = Path(tmpdir) / f"player_attributes_history_{season}.csv"
                download_with_resume(url=url, dest=tmp_csv)
                return _read_history_csv(tmp_csv)
        except RemoteError:
            logger.info("Not found at %s", url)
    return None


@lru_cache(maxsize=4)
def load_attributes_history(season: str) -> pd.DataFrame | None:
    """
    The per-day attributes history for a season, or None if there is none.

    A finished season's file never changes, so the packaged copy is read first and
    the download is the fallback. The current season's grows every day, so the
    download comes first and the packaged copy is the fallback - which is what
    keeps this working offline.
    """
    if not has_attributes_history(season):
        logger.info(
            "Player attributes history not available before %s season, skipping",
            ATTRIBUTES_HISTORY_FIRST_SEASON,
        )
        return None

    logger.info("Loading player attributes history for season %s", season)
    sources = (
        (_load_downloaded, _load_packaged)
        if season == CURRENT_SEASON
        else (_load_packaged, _load_downloaded)
    )
    for source in sources:
        df_attributes = source(season)
        if df_attributes is not None:
            return df_attributes

    logger.warning(
        "Could not load player attributes history for season %s from any known "
        "location",
        season,
    )
    return None


def _as_news(value: Any) -> str | None:
    """The news column of one history row, as the database wants it."""
    # `str100` column, and the longest news seen so far is 99 characters.
    return None if pd.isna(value) else str(value)[:100]


def _as_gameweek_or_chance(value: Any) -> int | None:
    """One of the two integer availability columns, as the database wants it."""
    # pandas reads a column with any blanks as float64, so these arrive as
    # `numpy.float64` and need narrowing back before they reach an INTEGER column.
    return None if pd.isna(value) else int(value)


def availability_from_row(row: "pd.Series[Any]") -> Availability:
    """The availability one row of the attributes history describes."""
    return Availability(
        news=_as_news(row["news"]),
        chance_of_playing_next_round=_as_gameweek_or_chance(
            row["chance_of_playing_next_round"]
        ),
        return_gameweek=_as_gameweek_or_chance(row["return_gameweek"]),
    )


@dataclass(frozen=True)
class AvailabilityIndex:
    """
    Availability by gameweek, for every player the history covers.

    Keyed by opta code and by name, the same two-step lookup
    `filter_attributes_for_player` does, so that one pass over the file serves
    every player instead of one boolean mask each.
    """

    by_opta_code: dict[tuple[str, int], Availability]
    by_name: dict[tuple[str, int], Availability]

    def get(self, player: Player, gameweek: int) -> Availability | None:
        """This player's availability in a gameweek, or None if not covered."""
        if player.opta_code is not None:
            return self.by_opta_code.get((player.opta_code, gameweek))
        return self.by_name.get((player.name, gameweek))


def get_availability_by_gameweek(
    player_attributes: pd.DataFrame, gameweek_dates: dict[int, datetime.date]
) -> AvailabilityIndex:
    """
    Every player's availability as of the first matchday of each gameweek.

    `gameweek_dates` maps a gameweek to the date of its earliest fixture. The
    daily snapshot is taken before that day's kickoffs and so before the
    gameweek's deadline, which is what makes `chance_of_playing_next_round` on
    that row refer to that gameweek.
    """
    gameweek_by_date = {day: gameweek for gameweek, day in gameweek_dates.items()}
    covered = player_attributes[player_attributes["day"].isin(gameweek_by_date)]

    by_opta_code: dict[tuple[str, int], Availability] = {}
    by_name: dict[tuple[str, int], Availability] = {}
    for _, row in covered.iterrows():
        gameweek = gameweek_by_date[row["day"]]
        availability = availability_from_row(row)
        if not pd.isna(row["opta_code"]):
            by_opta_code[(str(row["opta_code"]), gameweek)] = availability
        if not pd.isna(row["player"]):
            by_name[(str(row["player"]), gameweek)] = availability
    return AvailabilityIndex(by_opta_code=by_opta_code, by_name=by_name)


def filter_attributes_for_player(
    player: Player, player_attributes: pd.DataFrame
) -> pd.DataFrame:
    """The rows of the attributes history that belong to one player."""
    if (opta_code := player.opta_code) is not None:
        mask = player_attributes["opta_code"] == opta_code
    else:
        logger.warning("Player %s has no opta_code", player)
        mask = player_attributes["player"] == player.name
    return player_attributes.loc[mask]


def covers_date(date: datetime.date, player_attributes: pd.DataFrame) -> bool:
    """
    Whether the history has an answer for this player on this day.

    Distinct from that answer being "nothing to report": a day the dump covers on
    which a player is fine is a fact about them, where a day it does not cover is
    the caller's cue to look somewhere else.
    """
    return (
        date >= ATTRIBUTES_HISTORY_START
        and (player_attributes["day"] == date).sum() == 1
    )


def get_availability_on_date(
    date: datetime.date, player: Player, player_attributes: pd.DataFrame
) -> tuple[str | None, int | None]:
    """
    A player's news and chance of playing on one day, or `(None, None)`.

    Both are None when there is nothing to report rather than when something
    went wrong: the packaged attributes history only starts on 12 September 2025,
    and a day with no row - or more than one - is skipped with a warning.
    """
    if date < ATTRIBUTES_HISTORY_START:
        return None, None
    mask = player_attributes["day"] == date
    if mask.sum() != 1:
        logger.warning(
            "Found %s attributes for %s on %s, expected 1 so skipping",
            mask.sum(),
            player,
            date,
        )
        return None, None

    row = player_attributes.iloc[mask.argmax()]
    return _as_news(row["news"]), _as_gameweek_or_chance(
        row["chance_of_playing_next_round"]
    )
