"""
Fast team-model checks against the small seeded database.

The equivalents in test_score_predictions.py fit against two full seasons and
dominate the suite's runtime. These assert the same things - does it fit, does it
know every team, are the probabilities usable - on eight teams and 64 matches, so
the answers arrive on the default run rather than only under `-m slow`.

Parametrized over `TEAM_MODELS`, so adding a model to the table is all it takes to have
it fitted here. `SAMPLED_MODELS` below is the one place a model has to be named.
"""

import math

import numpy as np
import pytest
from sqlalchemy import select

from airsenal.db.models import PlayerPrediction
from airsenal.db.queries.fixtures import get_fixtures_for_gameweeks
from airsenal.prediction.evaluation import score_team_model
from airsenal.prediction.player_models import build_player_model
from airsenal.prediction.run import make_predictedscore_table
from airsenal.prediction.team_models import TEAM_MODELS, build_team_model
from airsenal.prediction.team_models.fitting import (
    fixture_probabilities,
    get_fitted_team_model,
)
from airsenal.prediction.team_models.scorelines import PoissonScorelines
from tests.e2e.conftest import FUTURE_GAMEWEEKS, PAST_SEASONS, SEASON, TEAMS

FIT_SEASON = PAST_SEASONS[-1]
FIT_GAMEWEEK = 8


# Returns samples rather than probabilities, by design, so the [0, 1] and
# sums-to-one assertions below do not apply to it.
SAMPLED_MODELS = {"random"}


@pytest.fixture(scope="module", params=sorted(TEAM_MODELS))
def fitted(request, pipeline_db):
    model = build_team_model(request.param)
    return request.param, get_fitted_team_model(
        FIT_GAMEWEEK, FIT_SEASON, pipeline_db, model=model
    )


def test_model_fits(fitted):
    _name, model = fitted
    assert model is not None


def test_model_knows_every_team(fitted):
    _name, model = fitted
    assert set(model.teams) >= set(TEAMS)


def test_score_probabilities_are_usable(fitted):
    name, model = fitted
    probabilities = np.asarray(
        model.predict_score_n_proba(np.arange(4), TEAMS[0], TEAMS[1], home=True)
    )
    assert probabilities.shape == (4,)
    assert all(math.isfinite(float(p)) for p in probabilities)
    if name not in SAMPLED_MODELS:
        assert all(0.0 <= float(p) <= 1.0 for p in probabilities)


# `predict_outcome_proba` is part of the TeamModel protocol, so every model in
# the table answers it - the null models included.
@pytest.mark.parametrize("name", sorted(TEAM_MODELS))
def test_fixture_probabilities_covers_every_fixture(pipeline_db, name):
    model = build_team_model(name)
    df = fixture_probabilities(
        FUTURE_GAMEWEEKS[0], PAST_SEASONS[-1], dbsession=pipeline_db, model=model
    )
    # four fixtures per gameweek in the seeded database
    assert len(df) == 4
    assert set(df.columns) >= {"home_team", "away_team"}


@pytest.mark.parametrize("name", ["constant", "random"])
def test_outcome_probabilities_sum_to_one(pipeline_db, name):
    """
    The null models convolve two independent goal counts, so they must.

    Named rather than taken from the table: this is a property of those two
    models, not of every team model.
    """
    df = fixture_probabilities(
        FUTURE_GAMEWEEKS[0],
        PAST_SEASONS[-1],
        dbsession=pipeline_db,
        model=build_team_model(name),
    )
    totals = (
        df["home_win_probability"] + df["draw_probability"] + df["away_win_probability"]
    )
    assert np.allclose(totals, 1.0)


# --- a model that predicts only a mean -----------------------------------------
#
# The seam phase 2 exists for: a model fitted to a continuous quantity has no
# distribution over goal *counts*, so it implements `ExpectedGoalsTeamModel` and
# reaches the points calculation wrapped in `PoissonScorelines`. This is what a
# `TEAM_MODELS` entry for one would look like.


class AverageGoalsModel:
    """Every team scores the league average, with a flat home advantage."""

    def __init__(self, home_advantage: float = 0.2):
        self.home_advantage = home_advantage
        self.teams: list[str] | None = None
        self.rate = 0.0

    def fit(self, training_data):
        self.teams = sorted(
            {str(t) for t in training_data["home_team"]}
            | {str(t) for t in training_data["away_team"]}
        )
        goals = np.concatenate(
            [training_data["home_goals"], training_data["away_goals"]]
        )
        self.rate = float(np.mean(goals))
        return self

    def add_new_team(self, team_name, **kwargs):
        del kwargs
        self.teams = sorted({*(self.teams or []), team_name})

    def predict_expected_goals(self, team, opponent, home=True, **kwargs):
        del team, opponent, kwargs
        return self.rate + (self.home_advantage if home else 0.0)


def _wrapped_average_goals():
    """What the table entry would be: the model, wrapped so it has scorelines."""
    return PoissonScorelines(AverageGoalsModel())


def test_a_mean_only_model_can_be_fitted(pipeline_db):
    """`get_fitted_team_model` takes it because the wrapper is a scoreline model."""
    model = get_fitted_team_model(
        FIT_GAMEWEEK, FIT_SEASON, pipeline_db, model=_wrapped_average_goals()
    )
    assert set(model.teams) >= set(TEAMS)
    assert model.predict_expected_goals(TEAMS[0], TEAMS[1]) > 0


def test_a_mean_only_model_reaches_the_points_calculation(pipeline_db):
    """
    The whole point of the phase: no distribution of its own, still predictable.

    A model with only `predict_expected_goals` produces real predicted points,
    without the points calculation knowing anything had been adapted.
    """
    tag = make_predictedscore_table(
        gameweeks=FUTURE_GAMEWEEKS[:1],
        season=SEASON,
        player_model=build_player_model("constant"),
        team_model=_wrapped_average_goals(),
        dbsession=pipeline_db,
    )
    points = pipeline_db.scalars(
        select(PlayerPrediction.predicted_points).where(PlayerPrediction.tag == tag)
    ).all()
    assert points
    assert all(math.isfinite(p) for p in points)
    assert any(p > 0 for p in points)


def test_a_mean_only_model_can_be_scored(pipeline_db):
    """And it is scored by the same function as every other team model."""
    model = get_fitted_team_model(
        FIT_GAMEWEEK, FIT_SEASON, pipeline_db, model=_wrapped_average_goals()
    )
    fixtures = get_fixtures_for_gameweeks(
        [FIT_GAMEWEEK], season=FIT_SEASON, dbsession=pipeline_db
    )
    score = score_team_model(model, fixtures)
    assert score.n_observations == len(fixtures)
    assert math.isfinite(score.total_log_probability)
