"""The PlayerScore rows the point-component fits are built from."""

import pandas as pd

from airsenal.db.queries.scores import get_player_scores_df
from tests.conftest import past_data_session_scope


def test_get_player_scores_df():
    """The row filter the bonus, save and card fits share: season, gameweek, minutes."""
    with past_data_session_scope() as ts:
        df = get_player_scores_df(season="1819", gameweek=12, dbsession=ts)
        # check type and columns
        assert len(df) > 0
        assert isinstance(df, pd.DataFrame)
        req_cols = [
            "player_id",
            "minutes",
            "saves",
            "bonus",
            "yellow_cards",
            "red_cards",
        ]
        for col in req_cols:
            assert col in df.columns
        # test player scores correctly filtered by gameweek and season
        for _, row in df.iterrows():
            assert row["season"] in ["1718", "1819"]
            if row["season"] == "1819":
                assert row["gameweek"] < 12
        # test filtering on min minutes
        df = get_player_scores_df(
            season="1819", gameweek=12, min_minutes=10, dbsession=ts
        )
        assert len(df) > 0
        assert all(df["minutes"] >= 10)
        # test filtering on max minutes
        df = get_player_scores_df(
            season="1819", gameweek=12, max_minutes=10, dbsession=ts
        )
        assert len(df) > 0
        assert all(df["minutes"] <= 10)
