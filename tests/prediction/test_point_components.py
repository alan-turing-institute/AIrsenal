"""
The fitted averages a predicted score adds to its event points.

Bonus, saves and cards are not modelled; they are each player's own average,
shrunk towards a group prior. `test_points.py` covers what the predictor does
with the numbers these produce.
"""

import pandas as pd

from airsenal.prediction.point_components import (
    fit_bonus_points,
    fit_card_points,
    fit_save_points,
    mean_group_prior,
)
from tests.conftest import past_data_session_scope


def test_mean_group_prior():
    """The empirical Bayes mean, with the prior weighted in."""
    df = pd.DataFrame(
        {
            "player_id": [1, 1, 1, 1, 2, 2],
            "bonus": [1, 1, 1, 1, 2, 2],
            "season": ["2526"] * 6,
            "gameweek": [1] * 6,
            "position": ["MID"] * 4 + ["FWD"] * 2,
        }
    )

    mean_1 = mean_group_prior(df, "player_id", "bonus", n_prior=0)
    assert mean_1.loc[1] == 1
    assert mean_1.loc[2] == 2

    n_prior = 6
    prior = 8 / 6
    mean_1_exp = (1 * 4 + n_prior * prior) / (4 + n_prior)
    mean_2_exp = (2 * 2 + n_prior * prior) / (2 + n_prior)
    mean_actual = mean_group_prior(df, "player_id", "bonus", n_prior=n_prior)
    assert mean_actual.loc[1] == mean_1_exp
    assert mean_actual.loc[2] == mean_2_exp

    mean_pos = mean_group_prior(
        df, "player_id", "bonus", n_prior=n_prior, prior_by_position=True
    )
    assert mean_pos.loc[1] == 1
    assert mean_pos.loc[2] == 2


def test_fit_bonus():
    with past_data_session_scope() as ts:
        df_bonus = fit_bonus_points(gameweek=1, season="1819", dbsession=ts)
        assert len(df_bonus) == 2
        for df in df_bonus:
            assert isinstance(df, pd.Series)
            assert len(df) > 0
            assert all(df <= 3)
            assert all(df >= 0)


def test_fit_saves():
    with past_data_session_scope() as ts:
        df_saves = fit_save_points(gameweek=1, season="1819", dbsession=ts)
        assert isinstance(df_saves, pd.Series)
        assert len(df_saves) > 0
        assert all(df_saves >= 0)


def test_fit_cards():
    with past_data_session_scope() as ts:
        df_cards = fit_card_points(gameweek=1, season="1819", dbsession=ts)
        assert isinstance(df_cards, pd.Series)
        assert len(df_cards) > 0
        assert all(df_cards <= 0)
        assert all(df_cards >= -3)
