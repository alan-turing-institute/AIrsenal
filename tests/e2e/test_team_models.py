"""
Fast team-model checks against the small seeded database.

The equivalents in test_score_predictions.py fit against two full seasons and
take 29 seconds between them - 60% of the whole suite. These assert the same
things (does it fit, does it know every team, are the probabilities usable) on
eight teams and 64 matches, so the answers arrive on every run rather than only
in the nightly job.

`random` is included, which the slow versions never covered at all.
"""

import math

import numpy as np
import pytest

from airsenal.prediction.models import build_team_model
from airsenal.prediction.team_models.dixon_coles import (
    fixture_probabilities,
    get_fitted_team_model,
)
from tests.e2e.conftest import FUTURE_GAMEWEEKS, PAST_SEASONS, TEAMS

FIT_SEASON = PAST_SEASONS[-1]
FIT_GAMEWEEK = 8


@pytest.fixture(scope="module", params=["extended", "neutral", "constant", "random"])
def fitted(request, pipeline_db):
    model = build_team_model(request.param)
    return request.param, get_fitted_team_model(
        FIT_SEASON, FIT_GAMEWEEK, pipeline_db, model=model
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
    if name != "random":
        # the random model returns samples, not probabilities, by design
        assert all(0.0 <= float(p) <= 1.0 for p in probabilities)


@pytest.mark.parametrize("name", ["extended", "neutral"])
def test_fixture_probabilities_covers_every_fixture(pipeline_db, name):
    model = build_team_model(name)
    df = fixture_probabilities(
        FUTURE_GAMEWEEKS[0], PAST_SEASONS[-1], dbsession=pipeline_db, model=model
    )
    # four fixtures per gameweek in the seeded database
    assert len(df) == 4
    assert set(df.columns) >= {"home_team", "away_team"}
