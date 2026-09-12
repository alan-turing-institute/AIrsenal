"""
Reading the per-day attributes history back.

The file has one row per player per day; what the ingest wants is one row per
player per gameweek, taken from the day that gameweek's first match was played.
"""

import datetime

import pandas as pd
import pytest

from airsenal.db.models import Player
from airsenal.ingest.attributes_history import (
    ATTRIBUTES_HISTORY_FIRST_SEASON,
    Availability,
    get_availability_by_gameweek,
    has_attributes_history,
    load_attributes_history,
)

GAMEWEEK_DATES = {
    1: datetime.date(2025, 9, 13),
    2: datetime.date(2025, 9, 20),
}


def _player(player_id=1, name="Bob", opta_code="p1"):
    player = Player()
    player.player_id = player_id
    player.name = name
    player.opta_code = opta_code
    return player


def _history(rows):
    """A history frame in the columns the reader uses."""
    frame = pd.DataFrame(
        rows,
        columns=[
            "day",
            "opta_code",
            "player",
            "news",
            "chance_of_playing_next_round",
            "return_gameweek",
        ],
    )
    # pandas reads a column with any blank as float64, which is how the real file
    # arrives and the reason the values need narrowing on the way out.
    frame["chance_of_playing_next_round"] = frame[
        "chance_of_playing_next_round"
    ].astype("float64")
    frame["return_gameweek"] = frame["return_gameweek"].astype("float64")
    return frame


def test_the_row_on_a_gameweeks_first_matchday_is_the_one_used():
    """The snapshot is taken before that day's kickoffs, so before the deadline."""
    frame = _history(
        [
            (datetime.date(2025, 9, 12), "p1", "Bob", "Knee injury", 0, 3),
            (GAMEWEEK_DATES[1], "p1", "Bob", "Knock", 25, None),
            (datetime.date(2025, 9, 14), "p1", "Bob", "Fine now", 100, None),
        ]
    )

    index = get_availability_by_gameweek(frame, GAMEWEEK_DATES)

    assert index.get(_player(), 1) == Availability(
        news="Knock", chance_of_playing_next_round=25, return_gameweek=None
    )


def test_a_gameweek_the_history_does_not_cover_has_no_answer():
    """The daily dump only started partway through 25/26, and skipped a few days."""
    frame = _history([(GAMEWEEK_DATES[1], "p1", "Bob", None, 0, 3)])

    index = get_availability_by_gameweek(frame, GAMEWEEK_DATES)

    assert index.get(_player(), 2) is None


def test_a_player_the_history_does_not_cover_has_no_answer():
    frame = _history([(GAMEWEEK_DATES[1], "p2", "Alice", None, 0, 3)])

    index = get_availability_by_gameweek(frame, GAMEWEEK_DATES)

    assert index.get(_player(), 1) is None


def test_an_empty_news_and_chance_come_back_as_none():
    """
    A blank cell is a float nan, and an integer column with any blank is float64.

    Both have to be narrowed before they reach a VARCHAR and an INTEGER column;
    a raw nan used to be written into `PlayerScore.news` as a float.
    """
    frame = _history([(GAMEWEEK_DATES[1], "p1", "Bob", None, None, None)])

    found = get_availability_by_gameweek(frame, GAMEWEEK_DATES).get(_player(), 1)

    assert found == Availability(
        news=None, chance_of_playing_next_round=None, return_gameweek=None
    )


def test_a_chance_of_playing_is_an_int_not_a_float():
    frame = _history([(GAMEWEEK_DATES[1], "p1", "Bob", "Knock", 50, 2)])

    found = get_availability_by_gameweek(frame, GAMEWEEK_DATES).get(_player(), 1)

    assert isinstance(found.chance_of_playing_next_round, int)
    assert isinstance(found.return_gameweek, int)


def test_a_player_with_no_opta_code_is_matched_by_name():
    """The fallback `filter_attributes_for_player` uses, in the bulk lookup too."""
    frame = _history([(GAMEWEEK_DATES[1], "p9", "Bob", "Knock", 0, None)])

    index = get_availability_by_gameweek(frame, GAMEWEEK_DATES)

    assert index.get(_player(opta_code=None), 1) is not None
    # the opta code is what wins when there is one, and this one does not match
    assert index.get(_player(), 1) is None


def test_news_is_truncated_to_the_column_width():
    frame = _history([(GAMEWEEK_DATES[1], "p1", "Bob", "x" * 150, 0, None)])

    found = get_availability_by_gameweek(frame, GAMEWEEK_DATES).get(_player(), 1)

    assert len(found.news) == 100


@pytest.mark.parametrize(
    ("season", "expected"), [("2425", False), ("2526", True), ("2627", True)]
)
def test_which_seasons_the_daily_dump_covers(season, expected):
    assert has_attributes_history(season) is expected


def test_a_season_before_the_dump_started_loads_nothing():
    assert load_attributes_history("2425") is None
    assert ATTRIBUTES_HISTORY_FIRST_SEASON == "2526"
