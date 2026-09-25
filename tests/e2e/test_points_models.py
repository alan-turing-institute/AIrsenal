"""
Every points model, against a real database.

Parametrized over `POINTS_MODELS`, so a new entry gets a real fit, a real
prediction and a real score without a test being written for it. Whatever a
model reports about how it got to its number is scored, and whatever it does not
report is not held against it.
"""

import math

import pytest

from airsenal.db.queries.fixtures import get_fixtures_for_gameweeks
from airsenal.db.queries.players import list_players
from airsenal.db.queries.scores import get_player_scores_for_gameweeks
from airsenal.prediction.evaluation import (
    backtest_breakdown,
    score_prediction_breakdown,
)
from airsenal.prediction.points_models import POINTS_MODELS, build_points_model
from airsenal.prediction.protocols import (
    PointsFitRequest,
    PointsPrediction,
    PointsRequest,
)
from tests.e2e.conftest import PAST_SEASONS

SEASON = PAST_SEASONS[-1]
GAMEWEEKS = [7, 8]


@pytest.fixture(scope="module", params=sorted(POINTS_MODELS))
def fitted(request, pipeline_db):
    model = build_points_model(request.param)
    return model.fit(
        PointsFitRequest(gameweeks=GAMEWEEKS, season=SEASON, dbsession=pipeline_db)
    )


def a_request(pipeline_db):
    player = list_players(season=SEASON, gameweek=GAMEWEEKS[0], dbsession=pipeline_db)[
        0
    ]
    fixture = next(
        f
        for f in get_fixtures_for_gameweeks(
            [GAMEWEEKS[0]], season=SEASON, dbsession=pipeline_db
        )
        if player.team(GAMEWEEKS[0], SEASON) in (f.home_team, f.away_team)
    )
    return PointsRequest(
        player=player,
        fixture=fixture,
        root_gameweek=GAMEWEEKS[0],
        season=SEASON,
        dbsession=pipeline_db,
    )


def test_every_points_model_predicts_a_finite_number(fitted, pipeline_db):
    prediction = fitted.predict(a_request(pipeline_db))
    assert isinstance(prediction, PointsPrediction)
    assert math.isfinite(prediction.expected_points)


def test_a_breakdown_adds_up_to_the_number_it_came_from(fitted, pipeline_db):
    """
    Whatever a model reports has to be consistent with what it predicted.

    Each component is its own expectation over the possible minutes, and an
    expectation of a sum is a sum of expectations, so this holds exactly rather
    than approximately.
    """
    prediction = fitted.predict(a_request(pipeline_db))
    if prediction.components is None:
        pytest.skip("this model does not report components")
    assert sum(prediction.components.values()) == pytest.approx(
        prediction.expected_points
    )


def test_an_unfitted_model_says_so(pipeline_db):
    model = build_points_model()
    with pytest.raises(RuntimeError, match="not been fitted"):
        model.predict(a_request(pipeline_db))


def test_the_breakdown_is_scored_part_by_part(fitted, pipeline_db):
    """The four evaluations in one call, for a model that reports all of them."""
    score = score_prediction_breakdown(
        fitted,
        get_player_scores_for_gameweeks(GAMEWEEKS, SEASON, dbsession=pipeline_db),
        root_gameweek=GAMEWEEKS[0],
        season=SEASON,
        dbsession=pipeline_db,
    )
    assert score.points.n_observations > 0
    assert math.isfinite(score.points.mean_absolute_error)

    assert score.minutes is not None
    assert score.minutes.n_observations == score.points.n_observations
    assert math.isfinite(score.minutes.mean_absolute_error)

    assert score.involvement is not None
    assert math.isfinite(score.involvement.mean_absolute_error_goals)

    assert score.components
    # the components a run predicts, minus the one this database has no
    # defensive contributions for
    assert set(score.components) >= {"appearance", "attacking", "defending"}
    for name, component in score.components.items():
        assert math.isfinite(component.mean_absolute_error), name


def test_a_model_that_reports_nothing_is_still_scored_on_its_total(pipeline_db):
    """
    The seam's promise, end to end against a real database.

    Nothing about this model decomposes, and it is scored all the same.
    """

    class FlatPointsModel:
        """Everyone scores two points, always."""

        def fit(self, request):
            del request
            return self

        def predict(self, request):
            del request
            return PointsPrediction(expected_points=2.0)

    score = score_prediction_breakdown(
        FlatPointsModel(),
        get_player_scores_for_gameweeks(GAMEWEEKS, SEASON, dbsession=pipeline_db),
        root_gameweek=GAMEWEEKS[0],
        season=SEASON,
        dbsession=pipeline_db,
    )
    assert score.points.n_observations > 0
    assert score.minutes is None
    assert score.involvement is None
    assert score.components is None


def test_the_backtest_walks_the_season_forward(pipeline_db):
    """And writes nothing, unlike `backtest_points`: it scores as it predicts."""
    score = backtest_breakdown(
        season=SEASON, dbsession=pipeline_db, gameweeks=GAMEWEEKS
    )
    assert score.points.n_observations > 0
    assert score.minutes is not None
    assert score.components
