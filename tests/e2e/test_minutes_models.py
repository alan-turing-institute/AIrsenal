"""
Every minutes model, against a real database.

The twin of test_team_models.py and test_player_models.py: parametrized over
`MINUTES_MODELS`, so a new entry gets a real prediction and a real score
without a test being written for it.
"""

import math

import pytest
from sqlalchemy import select

from airsenal.db.models import Player, PlayerScore
from airsenal.db.queries.scores import get_player_scores_for_gameweeks
from airsenal.prediction.evaluation import (
    MinutesScore,
    backtest_minutes_model,
    score_minutes_model,
)
from airsenal.prediction.minutes_models import MINUTES_MODELS
from airsenal.prediction.protocols import MinutesRequest
from tests.e2e.conftest import PAST_SEASONS

SEASON = PAST_SEASONS[-1]
PREDICT_FROM = 6
SCORE_GAMEWEEKS = [7, 8]


@pytest.mark.parametrize("name", sorted(MINUTES_MODELS))
def test_every_minutes_model_predicts_a_usable_distribution(pipeline_db, name):
    model = MINUTES_MODELS[name]()
    player = pipeline_db.scalars(select(Player)).first()
    distribution = model.predict(
        MinutesRequest(
            player=player,
            root_gameweek=PREDICT_FROM,
            fixture_gameweek=PREDICT_FROM,
            season=SEASON,
            n_gameweeks=len(SCORE_GAMEWEEKS),
            dbsession=pipeline_db,
        )
    )
    assert distribution.minutes
    assert sum(distribution.weights) == pytest.approx(1.0)
    # the seeded database gives everyone 45 or 90 minutes in every fixture
    assert 0.0 <= distribution.expected_minutes <= 90.0


@pytest.mark.parametrize("name", sorted(MINUTES_MODELS))
def test_every_minutes_model_can_be_scored(pipeline_db, name):
    player_scores = get_player_scores_for_gameweeks(
        SCORE_GAMEWEEKS, SEASON, dbsession=pipeline_db
    )
    score = score_minutes_model(
        MINUTES_MODELS[name](),
        player_scores,
        season=SEASON,
        dbsession=pipeline_db,
    )
    assert score.n_observations == len(player_scores)
    assert math.isfinite(score.mean_absolute_error)
    assert 0.0 <= score.band_accuracy <= 1.0
    assert 0.0 <= score.impossible_fraction <= 1.0
    # a log probability is never positive
    assert score.mean_log_probability <= 0


def test_the_backtest_walks_the_season_forward(pipeline_db):
    score = backtest_minutes_model(
        MINUTES_MODELS["recent"],
        season=SEASON,
        dbsession=pipeline_db,
        gameweeks=SCORE_GAMEWEEKS,
    )
    assert score.n_observations > 0
    assert math.isfinite(score.mean_absolute_error)


def test_a_player_who_always_plays_the_same_minutes_is_predicted_exactly(pipeline_db):
    """
    The seeded players are deterministic, so a perfect score is reachable.

    If this fails, the model is not reading the appearances it is given.
    """
    always_90 = pipeline_db.scalars(
        select(Player).where(Player.player_id % 5 != 0)
    ).first()
    scores = [
        score
        for score in get_player_scores_for_gameweeks(
            SCORE_GAMEWEEKS, SEASON, dbsession=pipeline_db
        )
        if score.player_id == always_90.player_id
    ]
    assert scores, "expected the chosen player to have played"
    assert all(score.minutes == 90 for score in scores)
    score = score_minutes_model(
        MINUTES_MODELS["recent"](), scores, season=SEASON, dbsession=pipeline_db
    )
    assert score.mean_absolute_error == pytest.approx(0.0)
    assert score.band_accuracy == 1.0


def test_an_empty_score_is_a_number_not_a_crash():
    empty = MinutesScore()
    assert empty.mean_absolute_error == 0.0
    assert empty.band_accuracy == 0.0
    assert empty.mean_log_probability == 0.0
    assert empty.impossible_fraction == 0.0


def test_a_band_the_model_never_sampled_is_recorded_as_impossible(pipeline_db):
    """
    The weakness of a sample standing in for a distribution.

    Every seeded appearance is 45 or 90 minutes, so the model gives a benching
    no weight at all - and the log score alone would not say why it was so bad.
    """
    score = pipeline_db.scalars(select(PlayerScore)).first()
    original = score.minutes
    score.minutes = 0
    try:
        result = score_minutes_model(
            MINUTES_MODELS["recent"](),
            [score],
            season=SEASON,
            dbsession=pipeline_db,
        )
    finally:
        score.minutes = original
    assert result.n_observations == 1
    assert result.n_impossible == 1
    assert result.impossible_fraction == 1.0


def test_a_performance_with_no_gameweek_is_skipped(pipeline_db):
    """A fixture with no gameweek cannot be predicted from anywhere."""
    score = pipeline_db.scalars(select(PlayerScore)).first()
    original = score.fixture.gameweek
    score.fixture.gameweek = None
    try:
        result = score_minutes_model(
            MINUTES_MODELS["recent"](),
            [score],
            season=SEASON,
            dbsession=pipeline_db,
        )
    finally:
        score.fixture.gameweek = original
    assert result.n_observations == 0
    assert result.n_skipped == 1
