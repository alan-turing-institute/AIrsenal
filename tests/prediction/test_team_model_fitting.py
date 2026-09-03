"""
Fitting a team model, and the frames it is fitted from.

As test_player_model_fitting.py: the table is in test_models.py and every entry
is fitted against the small seeded database in tests/e2e/test_team_models.py.
"""

import pandas as pd
import pytest
from bpl import ExtendedDixonColesMatchPredictor, NeutralDixonColesMatchPredictor

from airsenal.prediction.team_models.dixon_coles import (
    DEFAULT_TEAM_EPSILON,
    DixonColesTeamModel,
)
from airsenal.prediction.team_models.fitting import (
    fixture_probabilities,
    get_fitted_team_model,
    get_ratings_dict,
    get_result_dict,
)
from tests.conftest import past_data_session_scope


def test_get_result_dict():
    with past_data_session_scope() as ts:
        d = get_result_dict(10, "1819", ts)
        assert isinstance(d, dict)
        assert len(d) > 0


def test_get_ratings_dict():
    with past_data_session_scope() as ts:
        rd = get_result_dict(10, "1819", ts)
        teams = set(rd["home_team"]) | set(rd["away_team"])
        d = get_ratings_dict("1819", teams, ts)
        assert isinstance(d, dict)
        assert len(d) >= 20


@pytest.mark.slow
def test_get_fitted_team_model():
    """
    Fit every team model against two full seasons.

    22 seconds, and almost all of it is jax. The shape and coverage assertions
    are worth having on every run, so they are duplicated against the small e2e
    database in tests/e2e/test_team_models.py; this stays as the "does it still
    work on real data" check, and is marked `slow` so it runs in its own CI step.
    """
    # extended model
    with past_data_session_scope() as ts:
        model_team = get_fitted_team_model(10, "1819", ts, model=DixonColesTeamModel())
        assert isinstance(model_team.model, ExtendedDixonColesMatchPredictor)
    # extended model with default epsilon
    with past_data_session_scope() as ts:
        model_team = get_fitted_team_model(10, "1819", ts)
        assert isinstance(model_team.model, ExtendedDixonColesMatchPredictor)
        assert model_team.epsilon == DEFAULT_TEAM_EPSILON
    # extended model with epsilon = 0.5
    with past_data_session_scope() as ts:
        model_team = get_fitted_team_model(
            10, "1819", ts, model=DixonColesTeamModel(epsilon=0.5)
        )
        assert isinstance(model_team.model, ExtendedDixonColesMatchPredictor)
        assert model_team.epsilon == 0.5
    # neutral model with epsilon = 0.5
    with past_data_session_scope() as ts:
        model_team = get_fitted_team_model(
            10, "1819", ts, model=DixonColesTeamModel(neutral=True, epsilon=0.5)
        )
        assert isinstance(model_team.model, NeutralDixonColesMatchPredictor)
        assert model_team.epsilon == 0.5
    # neutral model with no epsilon passed
    with past_data_session_scope() as ts:
        model_team = get_fitted_team_model(
            10, "1819", ts, model=DixonColesTeamModel(neutral=True)
        )
        assert isinstance(model_team.model, NeutralDixonColesMatchPredictor)
        assert model_team.epsilon == DEFAULT_TEAM_EPSILON


@pytest.mark.slow
def test_fixture_probabilities():
    with past_data_session_scope() as ts:
        df = fixture_probabilities(20, "1819", dbsession=ts)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10
