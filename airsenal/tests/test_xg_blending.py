"""
Tests for blending expected goals/assists into the player model's observations.
"""

import numpy as np
import pandas as pd
import pytest

from airsenal.framework.player_model import ConjugatePlayerModel
from airsenal.framework.prediction_utils import blend_with_expected


def make_df():
    return pd.DataFrame(
        {
            "player_id": [1, 1, 2, 2],
            "goals": [1.0, 0.0, 0.0, 0.0],
            "assists": [0.0, 1.0, 0.0, 0.0],
            "expected_goals": [0.4, 0.2, 0.6, 0.5],
            "expected_assists": [0.1, 0.3, 0.2, 0.2],
        }
    )


def test_zero_weight_is_unchanged():
    df = make_df()
    out = blend_with_expected(df, xg_weight=0.0)
    pd.testing.assert_frame_equal(out, df)


def test_full_weight_uses_expected_only():
    out = blend_with_expected(make_df(), xg_weight=1.0)
    assert out["goals"].tolist() == [0.4, 0.2, 0.6, 0.5]
    assert out["assists"].tolist() == [0.1, 0.3, 0.2, 0.2]


def test_partial_weight_interpolates():
    out = blend_with_expected(make_df(), xg_weight=0.5)
    # (1 - 0.5) * 1.0 + 0.5 * 0.4
    assert out["goals"][0] == pytest.approx(0.7)
    # a player who scored nothing but had 0.6 xG gets credit for the chances
    assert out["goals"][2] == pytest.approx(0.3)


def test_missing_xg_falls_back_to_actuals():
    """FPL only published xG from 2022/23, and the zero-padding rows have none, so
    those matches must keep their actual values rather than becoming NaN."""
    df = make_df()
    df.loc[0, "expected_goals"] = np.nan
    df.loc[1, "expected_assists"] = None

    out = blend_with_expected(df, xg_weight=1.0)

    assert out["goals"][0] == 1.0, "no xG for this match, keep the actual goal"
    assert out["assists"][1] == 1.0, "no xA for this match, keep the actual assist"
    assert not out[["goals", "assists"]].isna().to_numpy().any()


def test_no_expected_columns_at_all():
    """An older database may not have the columns."""
    df = pd.DataFrame({"player_id": [1], "goals": [1.0], "assists": [0.0]})
    out = blend_with_expected(df, xg_weight=1.0)
    pd.testing.assert_frame_equal(out, df)


@pytest.mark.parametrize("weight", [-0.1, 1.5])
def test_invalid_weight_rejected(weight):
    with pytest.raises(ValueError, match="xg_weight must be between 0 and 1"):
        blend_with_expected(make_df(), xg_weight=weight)


def test_conjugate_model_accepts_fractional_involvements():
    """The conjugate model sums involvements into Dirichlet parameters, so it can
    take the fractional values blending produces - unlike the multinomial
    likelihood in NumpyroPlayerModel."""
    # 2 players, 3 matches, (goals, assists, neither)
    y = np.array(
        [
            [[0.4, 0.1, 1.5], [0.2, 0.3, 1.5], [0.0, 0.0, 0.0]],
            [[0.6, 0.2, 1.2], [0.5, 0.2, 1.3], [0.1, 0.1, 1.8]],
        ]
    )
    minutes = np.array([[90, 90, 0], [90, 60, 90]])

    model = ConjugatePlayerModel().fit(
        {"y": y, "minutes": minutes, "player_ids": np.array([1, 2])},
        epsilon=None,
    )
    probs = model.get_probs()

    assert len(probs["player_id"]) == 2
    for key in ("prob_score", "prob_assist", "prob_neither"):
        assert np.all(np.isfinite(probs[key])), key
        assert np.all(probs[key] >= 0), key
    totals = probs["prob_score"] + probs["prob_assist"] + probs["prob_neither"]
    assert np.allclose(totals, 1.0)
