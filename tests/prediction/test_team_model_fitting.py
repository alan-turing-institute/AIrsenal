"""
Fitting a team model, and the frames it is fitted from.

As test_player_model_fitting.py: the table is in test_models.py and every entry
is fitted against the small seeded database in tests/e2e/test_team_models.py.
"""

import pytest
from bpl import ExtendedDixonColesMatchPredictor, NeutralDixonColesMatchPredictor

from airsenal.prediction.team_models.dixon_coles import (
    DEFAULT_TEAM_EPSILON,
    DixonColesTeamModel,
)
from airsenal.prediction.team_models.fitting import (
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
    Fit the goals-based team models against two full seasons.

    The "does it still work on real data" check, marked `slow` so it runs in its
    own CI step. tests/e2e/test_team_models.py asserts the shape and coverage
    against the small e2e database on every run.

    Every model is named, the default one included: this database is 1718 and
    1819, and the default model is fitted to expected goals, which the FPL API
    has only recorded since 2223. That is
    `test_the_default_model_needs_a_season_with_expected_goals` below.
    """
    # extended model, with the epsilon it defaults to
    with past_data_session_scope() as ts:
        model_team = get_fitted_team_model(10, "1819", ts, model=DixonColesTeamModel())
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


def test_the_default_model_needs_a_season_with_expected_goals():
    """
    The default model refuses a season with no expected goals.

    `XGTeamModel` refuses training data with no expected goals in it, which is
    every season before 2223, so a backtest or a replay of one has to name a
    model fitted to goals. Cheap because the refusal comes before any fitting.
    """
    with past_data_session_scope() as ts, pytest.raises(ValueError, match="No match"):
        get_fitted_team_model(10, "1819", ts)
