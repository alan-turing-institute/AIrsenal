"""
Fast player-model checks against the small seeded database.

The team-model twin of `test_team_models.py`, and the answer to how
`NumpyroPlayerModel` shipped for a release unable to fit at all: the only test
that fitted it ran against two full seasons, was marked xfail, and so said
nothing on any ordinary run.

Parametrized over `PLAYER_MODELS`, so adding a model to the table is all it takes
to have it fitted here. `build_player_model_for_test` in the conftest is the one place a
model is named, and only to shrink its sampling - not to skip it.
"""

import numpy as np
import pandas as pd
import pytest

from airsenal.game.enums import Position
from airsenal.prediction.player_models import PLAYER_MODELS
from airsenal.prediction.player_models.fitting import (
    fit_player_data,
    get_all_fitted_player_data,
)
from tests.e2e.conftest import (
    GAMEWEEKS_PER_PAST_SEASON,
    PAST_SEASONS,
    build_player_model_for_test,
)

FIT_SEASON = PAST_SEASONS[-1]
FIT_GAMEWEEK = GAMEWEEKS_PER_PAST_SEASON
PROBABILITY_COLUMNS = ["prob_score", "prob_assist", "prob_neither"]


@pytest.fixture(scope="module", params=sorted(PLAYER_MODELS))
def fitted(request, pipeline_db):
    model = build_player_model_for_test(request.param)
    return request.param, fit_player_data(
        Position.FWD, FIT_SEASON, FIT_GAMEWEEK, model=model, dbsession=pipeline_db
    )


def test_model_fits(fitted):
    _name, df = fitted
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_every_player_gets_all_three_probabilities(fitted):
    _name, df = fitted
    assert set(df.columns) >= set(PROBABILITY_COLUMNS)
    assert df[PROBABILITY_COLUMNS].notna().all().all()


def test_the_three_outcomes_partition_a_goal(fitted):
    """Score, assist and neither are exhaustive, so they sum to one per player."""
    _name, df = fitted
    assert np.allclose(df[PROBABILITY_COLUMNS].sum(axis=1), 1.0, atol=1e-5)
    assert (df[PROBABILITY_COLUMNS] >= 0).all().all()


@pytest.mark.parametrize("name", sorted(PLAYER_MODELS))
def test_every_position_is_fitted(pipeline_db, name):
    """`get_all_fitted_player_data` is what the pipeline calls, once per position."""
    data = get_all_fitted_player_data(
        FIT_SEASON,
        FIT_GAMEWEEK,
        model=build_player_model_for_test(name),
        dbsession=pipeline_db,
    )
    assert set(data) == {str(p) for p in Position}
    for position, df in data.items():
        assert len(df) > 0, f"no players fitted for {position}"
