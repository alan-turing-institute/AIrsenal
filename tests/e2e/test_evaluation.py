"""
Scoring a model against what actually happened.

The check that "is my new model better?" is answerable at all: every model in
the tables is scored the same way, on the same observations, and the scores
order the way the models do. The constant models are the floor - they know
nothing about who is playing - so a real model has to beat them.
"""

import math

import pytest
from sqlalchemy import select

from airsenal.db.models import PlayerScore
from airsenal.db.queries.fixtures import get_fixtures_for_gameweeks
from airsenal.game.enums import Position
from airsenal.prediction.evaluation import (
    ModelScore,
    backtest_team_model,
    player_outcome_probability,
    score_player_model,
    score_team_model,
)
from airsenal.prediction.player_models import PLAYER_MODELS
from airsenal.prediction.player_models.fitting import fit_player_data
from airsenal.prediction.team_models import TEAM_MODELS, build_team_model
from airsenal.prediction.team_models.fitting import get_fitted_team_model
from tests.e2e.conftest import PAST_SEASONS, build_player_model_for_test

SEASON = PAST_SEASONS[-1]
FIT_GAMEWEEK = 6
SCORE_GAMEWEEKS = [7, 8]


def test_an_empty_score_is_a_number_not_a_crash():
    empty = ModelScore()
    assert empty.mean_log_probability == 0.0


def test_scores_add():
    total = ModelScore(-4.0, 2) + ModelScore(-6.0, 4, n_skipped=1)
    assert total == ModelScore(-10.0, 6, 1)
    assert total.mean_log_probability == pytest.approx(-10.0 / 6)


@pytest.mark.parametrize("name", sorted(TEAM_MODELS))
def test_every_team_model_can_be_scored(pipeline_db, name):
    """The point of the module: one call per model, whatever the model is."""
    model = get_fitted_team_model(
        SEASON, FIT_GAMEWEEK, pipeline_db, model=build_team_model(name)
    )
    fixtures = get_fixtures_for_gameweeks(
        SCORE_GAMEWEEKS, season=SEASON, dbsession=pipeline_db
    )
    score = score_team_model(model, fixtures)
    assert score.n_observations == len(fixtures)
    assert math.isfinite(score.total_log_probability)
    # a log probability is never positive
    assert score.total_log_probability <= 0


def test_a_fitted_model_beats_the_constant_one(pipeline_db):
    """
    A fitted model should beat one that knows nothing about who is playing.

    If this stops holding, the score is not measuring anything.
    """
    fixtures = get_fixtures_for_gameweeks(
        SCORE_GAMEWEEKS, season=SEASON, dbsession=pipeline_db
    )
    scores = {}
    for name in ("extended", "constant"):
        model = get_fitted_team_model(
            SEASON, FIT_GAMEWEEK, pipeline_db, model=build_team_model(name)
        )
        scores[name] = score_team_model(model, fixtures).mean_log_probability
    assert scores["extended"] > scores["constant"]


def test_fixtures_without_a_result_are_skipped_not_scored(pipeline_db):
    """A future gameweek has no result, so there is nothing to score against."""
    model = get_fitted_team_model(
        SEASON, FIT_GAMEWEEK, pipeline_db, model=build_team_model("constant")
    )
    future = get_fixtures_for_gameweeks([1, 2], season="2526", dbsession=pipeline_db)
    score = score_team_model(model, future)
    assert score.n_observations == 0
    assert score.n_skipped == len(future)


def test_backtest_walks_the_season_forward(pipeline_db):
    """`TEAM_MODELS` entries are factories, so a table entry can be backtested."""
    score = backtest_team_model(
        TEAM_MODELS["constant"],
        season=SEASON,
        dbsession=pipeline_db,
        gameweeks=[6, 7, 8],
    )
    assert score.n_observations > 0
    assert math.isfinite(score.mean_log_probability)


@pytest.mark.parametrize("name", sorted(PLAYER_MODELS))
def test_every_player_model_can_be_scored(pipeline_db, name):
    probabilities = fit_player_data(
        Position.FWD,
        SEASON,
        FIT_GAMEWEEK,
        model=build_player_model_for_test(name),
        dbsession=pipeline_db,
    )
    player_scores = pipeline_db.scalars(select(PlayerScore)).all()
    score = score_player_model(probabilities, player_scores)
    assert score.n_observations > 0
    assert math.isfinite(score.total_log_probability)
    assert score.total_log_probability <= 0


def test_a_player_who_did_not_play_says_nothing_about_the_model():
    assert (
        player_outcome_probability(
            goals=0, assists=0, team_goals=2, minutes=0, probabilities=[0.1, 0.1, 0.8]
        )
        == 1.0
    )


def test_a_goalless_match_says_nothing_about_the_model():
    assert (
        player_outcome_probability(
            goals=0, assists=0, team_goals=0, minutes=90, probabilities=[0.1, 0.1, 0.8]
        )
        == 1.0
    )


def test_scoring_a_goal_is_likelier_for_a_better_scorer():
    """The whole point of a player model, so the scorer has to reflect it."""
    likely = player_outcome_probability(
        goals=1, assists=0, team_goals=1, minutes=90, probabilities=[0.5, 0.1, 0.4]
    )
    unlikely = player_outcome_probability(
        goals=1, assists=0, team_goals=1, minutes=90, probabilities=[0.05, 0.1, 0.85]
    )
    assert likely > unlikely


def test_more_goals_than_the_team_scored_is_not_scored():
    """An own goal makes 'neither' negative; it is not an outcome the model claims."""
    assert (
        player_outcome_probability(
            goals=2, assists=1, team_goals=1, minutes=90, probabilities=[0.3, 0.3, 0.4]
        )
        == 1.0
    )
