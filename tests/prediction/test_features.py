"""The player-level frame the models are fitted on."""

from sqlalchemy import select

from airsenal.db.models import Fixture, Result
from airsenal.prediction.features import get_player_history_df
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
