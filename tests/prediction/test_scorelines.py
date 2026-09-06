"""
A team model that predicts only a mean, made usable by the points calculation.

`PoissonScorelines` is the seam that lets a model with no distribution over goal
counts - an xG model fitted to a continuous quantity - be a `TEAM_MODELS` entry.
`tests/e2e/test_team_models.py` puts one through a real fit and prediction.
"""

import math
from typing import Any

import numpy as np
import pytest
from scipy.stats import poisson

from airsenal.game.scoring import MAX_GOALS
from airsenal.prediction.protocols import TeamFitData
from airsenal.prediction.team_models.scorelines import (
    PoissonScorelines,
    outcome_proba_from_scores,
)

TEAMS = ["AAA", "BBB"]


class FlatExpectedGoals:
    """The smallest `ExpectedGoalsTeamModel`: one rate, whoever is playing."""

    def __init__(self, rate: float = 1.4, home_advantage: float = 0.3):
        self.rate = rate
        self.home_advantage = home_advantage
        self.teams: list[str] | None = None
        self.fitted_with: TeamFitData | None = None

    def fit(self, training_data: TeamFitData) -> "FlatExpectedGoals":
        self.fitted_with = training_data
        self.teams = sorted(TEAMS)
        return self

    def add_new_team(self, team_name: str, **kwargs: Any) -> None:
        del kwargs
        self.teams = sorted({*(self.teams or []), team_name})

    def predict_expected_goals(
        self, team: str, opponent: str, home: bool = True, **kwargs: Any
    ) -> float:
        del team, opponent, kwargs
        return self.rate + (self.home_advantage if home else 0.0)


@pytest.fixture
def wrapped():
    return PoissonScorelines(FlatExpectedGoals())


def test_the_distribution_sums_to_one(wrapped):
    """
    Truncating the support must not lose probability.

    `score_team_model` scores a freak result against the last bin, so that bin
    has to carry the whole tail rather than only its own count.
    """
    goals = np.arange(MAX_GOALS + 1)
    probabilities = wrapped.predict_score_n_proba(goals, TEAMS[0], TEAMS[1])
    assert probabilities.sum() == pytest.approx(1.0)
    assert all(0.0 <= float(p) <= 1.0 for p in probabilities)


def test_the_last_count_carries_the_tail(wrapped):
    """So a 10-0 is unlikely rather than impossible."""
    rate = wrapped.predict_expected_goals(TEAMS[0], TEAMS[1])
    last = wrapped.predict_score_n_proba(np.array([MAX_GOALS]), TEAMS[0], TEAMS[1])
    assert float(last[0]) > float(poisson.pmf(MAX_GOALS, rate))
    assert float(last[0]) == pytest.approx(float(poisson.sf(MAX_GOALS - 1, rate)))


def test_the_mean_survives_the_wrapping(wrapped):
    """The point of a Poisson: its mean is the rate it was given."""
    goals = np.arange(MAX_GOALS + 1)
    probabilities = wrapped.predict_score_n_proba(goals, TEAMS[0], TEAMS[1])
    expected = float((goals * probabilities).sum())
    assert expected == pytest.approx(
        wrapped.predict_expected_goals(TEAMS[0], TEAMS[1]), rel=1e-3
    )


def test_negative_goal_counts_have_no_probability(wrapped):
    probabilities = wrapped.predict_score_n_proba(
        np.array([-1, 0, 1]), TEAMS[0], TEAMS[1]
    )
    assert float(probabilities[0]) == 0.0
    assert float(probabilities[1]) > 0.0


def test_home_advantage_reaches_the_distribution(wrapped):
    """The wrapper must not flatten what the inner model says about the fixture."""
    at_home = wrapped.predict_score_n_proba(
        np.arange(MAX_GOALS + 1), TEAMS[0], TEAMS[1], home=True
    )
    away = wrapped.predict_score_n_proba(
        np.arange(MAX_GOALS + 1), TEAMS[0], TEAMS[1], home=False
    )
    goals = np.arange(MAX_GOALS + 1)
    assert float((goals * at_home).sum()) > float((goals * away).sum())


def test_fitting_and_new_teams_reach_the_wrapped_model(wrapped):
    """The wrapper is transparent for everything except what it adds."""
    training_data = {"home_team": np.array(TEAMS), "away_team": np.array(TEAMS[::-1])}
    assert wrapped.fit(training_data) is wrapped
    assert wrapped.model.fitted_with is training_data
    assert wrapped.teams == sorted(TEAMS)
    wrapped.add_new_team("CCC")
    assert "CCC" in wrapped.teams


def test_outcome_probabilities_are_derived_not_required(wrapped):
    """
    A mean-only model gets win/draw/loss for free once it has a distribution.

    Which is why `predict_outcome_proba` is on the protocol rather than fetched
    with getattr - see the reasoning in commit 877cc945.
    """
    outcomes = wrapped.predict_outcome_proba([TEAMS[0]], [TEAMS[1]])
    total = outcomes["home_win"][0] + outcomes["draw"][0] + outcomes["away_win"][0]
    assert total == pytest.approx(1.0, abs=1e-6)
    # the home side is the one with the advantage
    assert outcomes["home_win"][0] > outcomes["away_win"][0]


def test_the_helper_and_the_method_agree(wrapped):
    direct = outcome_proba_from_scores(wrapped, [TEAMS[0]], [TEAMS[1]])
    through = wrapped.predict_outcome_proba([TEAMS[0]], [TEAMS[1]])
    for key in direct:
        assert math.isclose(float(direct[key][0]), float(through[key][0]))
