"""
Fitting a player model, and the minute-scaling it fits on.

The table itself is in test_models.py; every entry of it is fitted against the
small seeded database in tests/e2e/test_player_models.py. What is here is the
arithmetic - scaling and the conjugate posterior - plus the one `slow` fit
against two full seasons of real data.
"""

import numpy as np
import pandas as pd
import pytest

from airsenal.prediction.player_models import (
    ConjugatePlayerConfig,
    ConjugatePlayerModel,
    NumpyroPlayerModel,
)
from airsenal.prediction.player_models.fitting import fit_player_data
from airsenal.prediction.player_models.scaling import scale_goals_by_minutes
from tests.conftest import past_data_session_scope


def test_scale_goals_by_minutes():
    """
    Scaling goal involvements by minutes played shrinks the "neither" count.

    It falls by the fraction of minutes the player was not on the pitch for.
    """
    goals = np.zeros((2, 2, 3))
    goals[0, :, :] = np.array([[0, 0, 0], [1, 2, 3]])
    goals[1, :, :] = np.array([[0, 1, 2], [1, 0, 2]])
    minutes = np.array([[90, 90], [45, 45]])
    scaled_goals = scale_goals_by_minutes(goals, minutes)
    assert (scaled_goals == np.array([[1, 2, 3], [1, 1, 1]])).all()


def test_a_player_in_no_goalless_match_scales_to_nothing():
    """
    A player who was never on the pitch for a goal contributes no counts.

    Otherwise their zero-match average would be a division by zero. Only the real
    two-season fit used to reach this, so it is asserted directly here.
    """
    goals = np.zeros((2, 2, 3))
    goals[0, :, :] = np.array([[1, 0, 0], [0, 1, 0]])  # involved in two goals
    goals[1, :, :] = np.array([[0, 0, 0], [0, 0, 0]])  # never on for one
    minutes = np.array([[90, 90], [90, 90]])

    scaled_goals = scale_goals_by_minutes(goals, minutes)

    assert (scaled_goals[1] == np.array([0, 0, 0])).all()
    assert scaled_goals[0].sum() > 0


def test_get_conjugate_prior():
    pm = ConjugatePlayerModel(ConjugatePlayerConfig(n_goals_prior=0, epsilon=None))
    goals = np.zeros((2, 2, 3))
    goals[0, :, :] = np.array([[0, 0, 0], [2, 2, 5]])
    goals[1, :, :] = np.array([[0, 1, 2], [1, 0, 2]])

    minutes = np.array([[90, 90], [90, 90]])
    scaled_goals = scale_goals_by_minutes(goals, minutes)
    assert (pm.get_prior(scaled_goals, n_goals_prior=15) == np.array([3, 3, 9])).all()

    minutes = np.array([[90, 90], [45, 45]])
    scaled_goals = scale_goals_by_minutes(goals, minutes)
    assert (pm.get_prior(scaled_goals, n_goals_prior=4) == np.array([1, 1, 2])).all()


def test_fit_conjugate_player_model():
    """Fitting ConjugatePlayerModel gives the expected posterior."""
    pm = ConjugatePlayerModel(ConjugatePlayerConfig(n_goals_prior=0, epsilon=None))
    y = np.zeros((2, 2, 3))
    y[0, :, :] = np.array([[0, 0, 0], [1, 2, 3]])  # all y add to 4
    y[1, :, :] = np.array([[1, 2, 1], [2, 0, 0]])
    data = {
        "y": y,
        "player_ids": [0, 1],
        "minutes": 90 * np.ones((2, 2)),
    }

    pm = pm.fit(data)
    assert (pm.posterior == np.array([[1, 2, 3], [3, 2, 1]])).all()

    # A different prior means a different model object, not a different fit call.
    pm = ConjugatePlayerModel(ConjugatePlayerConfig(n_goals_prior=3, epsilon=None))
    pm = pm.fit(data)
    assert (pm.posterior == np.array([[2, 3, 4], [4, 3, 2]])).all()


@pytest.mark.slow
@pytest.mark.parametrize("model", [NumpyroPlayerModel(), ConjugatePlayerModel()])
def test_get_fitted_player_model(model):
    """
    Fit a player model against two full seasons.

    As `test_get_fitted_team_model` below: the shape and coverage assertions are
    worth having on every run, so they are duplicated against the small e2e
    database in tests/e2e/test_player_models.py, which asserts more of them and
    over every entry of the table. This stays as the "does it still work on real
    data" check, and is marked `slow` so it runs in its own CI step.
    """
    with past_data_session_scope() as ts:
        fitted = fit_player_data("FWD", 12, "1819", model=model, dbsession=ts)
        assert isinstance(fitted, pd.DataFrame)
        assert len(fitted) > 0
        # The three outcomes partition a goal, so they must sum to one per player.
        # Fitting at all is not enough: numpyro spent a release returning nothing
        # because it could not initialise.
        probabilities = fitted[["prob_score", "prob_assist", "prob_neither"]]
        assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
        assert (probabilities >= 0).all().all()
