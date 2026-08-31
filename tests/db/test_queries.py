"""The database query helpers: looking a player or a gameweek up."""

import pytest

from airsenal.db.models import Player
from airsenal.db.queries.gameweeks import (
    get_gameweek_by_date,
    get_gameweeks_array,
    get_last_complete_gameweek_in_db,
    get_return_gameweek_by_date,
)
from airsenal.db.queries.players import get_player, get_player_id, get_player_name
from airsenal.db.queries.scores import (
    get_last_complete_gameweek_of_player_scores_in_db,
)
from tests.conftest import TEST_PAST_SEASON, past_data_session_scope, session_scope


def test_get_player_name(fill_players):
    """A player can be looked up by id."""
    with session_scope() as tsession:
        assert get_player_name(1, tsession) == "Bob"


def test_get_player_id(fill_players):
    """A player can be looked up by name."""
    with session_scope() as tsession:
        assert get_player_id("Bob", tsession) == 1


def test_get_player(fill_players):
    """The same player comes back whether asked for by name or by id."""
    with session_scope() as tsession:
        p = get_player("Bob", tsession)
        assert isinstance(p, Player)
        assert p.player_id == 1


def test_get_return_gameweek_by_date():
    with past_data_session_scope() as ts:
        gw = get_return_gameweek_by_date(
            "2020-09-18", "ARS", season=TEST_PAST_SEASON, dbsession=ts
        )
        assert gw == 2

        gw = get_return_gameweek_by_date(
            "2020-09-20T12:34:00Z", "ARS", season=TEST_PAST_SEASON, dbsession=ts
        )
        assert gw == 3


def test_get_gameweek_by_date():
    with past_data_session_scope() as ts:
        gw = get_gameweek_by_date(
            "2020-09-20T12:34:00Z", season=TEST_PAST_SEASON, dbsession=ts
        )
        assert gw == 2


def test_get_last_complete_gameweek_in_db():
    with past_data_session_scope() as ts:
        gw = get_last_complete_gameweek_in_db(season=TEST_PAST_SEASON, dbsession=ts)
        assert gw == 5


def test_player_scores_have_their_own_high_water_mark():
    """
    Player scores are tracked separately from results, and can lag behind them.

    They are filled by a separate call that commits separately, so a failure
    between the two leaves the scores behind. The 2021 season in the test
    database is exactly that shape - results up to gameweek 5, no player scores
    at all - and a results-derived mark would call it up to date.
    """
    with past_data_session_scope() as ts:
        assert (
            get_last_complete_gameweek_in_db(season=TEST_PAST_SEASON, dbsession=ts) == 5
        )
        assert (
            get_last_complete_gameweek_of_player_scores_in_db(
                season=TEST_PAST_SEASON, dbsession=ts
            )
            == 0
        )


def test_player_scores_high_water_mark_matches_results_when_complete():
    """A season whose scores are all present is level with its results."""
    with past_data_session_scope() as ts:
        assert get_last_complete_gameweek_of_player_scores_in_db(
            season="1718", dbsession=ts
        ) == get_last_complete_gameweek_in_db(season="1718", dbsession=ts)


class TestGetGameweeksArrayIsToldTheWindow:
    """
    Neither a length nor an end gameweek is an error, not a default.

    How far ahead to look is a decision about a run, not about the gameweek
    table, so this function refuses to make it.
    """

    def test_a_window_with_neither_a_length_nor_an_end_is_refused(self):
        with (
            past_data_session_scope() as ts,
            pytest.raises(RuntimeError, match="Specify"),
        ):
            get_gameweeks_array(gameweek_start=1, season=TEST_PAST_SEASON, dbsession=ts)

    def test_a_length_is_enough(self):
        with past_data_session_scope() as ts:
            assert get_gameweeks_array(
                n_gameweeks=3, gameweek_start=1, season=TEST_PAST_SEASON, dbsession=ts
            ) == [1, 2, 3]

    def test_an_end_is_enough(self):
        """And it is inclusive, as `--gameweek-end` says it is."""
        with past_data_session_scope() as ts:
            assert get_gameweeks_array(
                gameweek_start=1,
                gameweek_end=4,
                season=TEST_PAST_SEASON,
                dbsession=ts,
            ) == [1, 2, 3, 4]

    def test_a_length_and_the_same_window_named_by_its_ends_agree(self):
        """
        The invariant the exclusive end broke.

        `--gameweek-end` says "Last gameweek to cover" and `airsenal replay`
        takes it that way, so three gameweeks from gameweek 1 has to mean the
        same thing whichever way a command names it.
        """
        with past_data_session_scope() as ts:
            by_length = get_gameweeks_array(
                n_gameweeks=3, gameweek_start=1, season=TEST_PAST_SEASON, dbsession=ts
            )
            by_ends = get_gameweeks_array(
                gameweek_start=1,
                gameweek_end=3,
                season=TEST_PAST_SEASON,
                dbsession=ts,
            )
        assert by_length == by_ends == [1, 2, 3]

    def test_both_at_once_is_still_refused(self):
        with (
            past_data_session_scope() as ts,
            pytest.raises(RuntimeError, match="Only one"),
        ):
            get_gameweeks_array(
                n_gameweeks=3,
                gameweek_start=1,
                gameweek_end=4,
                season=TEST_PAST_SEASON,
                dbsession=ts,
            )
