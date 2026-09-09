"""The player-level frame the models are fitted on."""

from sqlalchemy import select

from airsenal.db.models import Fixture, Result
from airsenal.prediction.features import (
    PLAYER_HISTORY_COLUMNS,
    blank_player_row,
    get_player_history_df,
)
from tests.conftest import past_data_session_scope


def test_get_player_history_df():
    """Only gameweeks up to the one asked for are considered."""
    with past_data_session_scope() as ts:
        df = get_player_history_df(season="1819", gameweek=12, dbsession=ts)
        assert len(df) > 0
        result_ids = df.match_id.unique()
        for result_id in result_ids:
            if result_id == 0:
                continue
            result = ts.scalars(
                select(Result).where(Result.result_id == int(result_id)).limit(1)
            )
            result_row = result.first()
            assert result_row is not None
            fixture_id = result_row.fixture_id
            fixture = ts.scalars(
                select(Fixture).where(Fixture.fixture_id == fixture_id).limit(1)
            ).first()
            assert fixture is not None
            assert fixture.season in ["1718", "1819"]
            if fixture.season == "1819":
                assert fixture.gameweek is not None
                assert fixture.gameweek < 12


def test_a_padding_row_covers_every_column():
    """
    The frame is built from dicts, so a row that misses a column becomes `nan`.

    `blank_player_row` derives its zeros from `PLAYER_HISTORY_COLUMNS`, which is
    what makes adding a column safe: the three parallel positional lists this
    replaced had to be kept in step by hand, and the xG work extended all three.
    """
    row = blank_player_row(player_id=7, player_name="A Player")
    assert set(row) == set(PLAYER_HISTORY_COLUMNS)


def test_a_padding_row_cannot_be_mistaken_for_a_performance():
    """
    Zero everywhere but the player's identity, which is how it is recognised.

    `get_empirical_bayes_estimates` drops rows by `match_id == 0` and a model
    fitted to expected goals excludes them by `team_expected_goals` of zero, so
    both tests have to keep finding one.
    """
    row = blank_player_row(player_id=7, player_name="A Player")
    assert (row["player_id"], row["player_name"]) == (7, "A Player")
    assert row["match_id"] == 0
    assert row["team_expected_goals"] == 0
    assert row["minutes"] == 0
    assert row["absence_reason"] is None
    assert row["absence_detail"] is None


def test_the_frame_has_the_columns_it_says_it_has():
    """The order is fixed, so a caller reading positionally is not surprised."""
    with past_data_session_scope() as ts:
        df = get_player_history_df(season="1819", gameweek=12, dbsession=ts)
        assert list(df.columns) == list(PLAYER_HISTORY_COLUMNS)
