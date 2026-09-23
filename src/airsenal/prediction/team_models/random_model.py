"""
A team model whose predictions are random but well-formed.

Used as a control in `airsenal replay`: a season played on random scorelines is
the floor a real model has to clear.
"""

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from airsenal.core.lookup import ConfigError
from airsenal.game.scoring import MAX_GOALS
from airsenal.prediction.protocols import TeamFitData
from airsenal.prediction.team_models.constant import teams_in, with_team
from airsenal.prediction.team_models.scorelines import (
    outcome_proba_from_scores,
)


class RandomTeamModel:
    """Random scoreline probabilities."""

    def __init__(
        self,
        max_goals: int = MAX_GOALS,
        random_state: int = 42,
        *,
        epsilon: float | None = None,
    ) -> None:
        if epsilon is not None:
            msg = "the random team model has no time weighting, so no epsilon"
            raise ConfigError(msg)
        self.max_goals = max_goals
        self.teams: list[str] | None = None
        self.rng = np.random.default_rng(random_state)
        # per-team probability of scoring 0..max_goals, drawn once at fit time
        self._goal_probabilities: dict[str, np.ndarray] = {}

    def _draw(self) -> np.ndarray:
        """A random probability vector over 0..max_goals."""
        weights = self.rng.random(self.max_goals + 1)
        return weights / weights.sum()

    def fit(self, training_data: TeamFitData) -> "RandomTeamModel":
        self.teams = teams_in(training_data)
        if not self.teams:
            msg = "No teams found in training data."
            raise ValueError(msg)
        self._goal_probabilities = {team: self._draw() for team in self.teams}
        return self

    def add_new_team(self, team_name: str, **kwargs: Any) -> None:
        del kwargs
        if self.teams is not None and team_name in self.teams:
            return
        self.teams = with_team(self.teams, team_name)
        self._goal_probabilities[team_name] = self._draw()

    def predict_score_n_proba(
        self,
        n: int | Iterable[int],
        team: str | Iterable[str],
        opponent: str | Iterable[str] = "",
        home: bool | None = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Probability that `team` scores each of `n` goals, whoever it plays."""
        del opponent, home, kwargs
        team_name = team if isinstance(team, str) else next(iter(team))
        probabilities = self._goal_probabilities.get(str(team_name))
        if probabilities is None:
            probabilities = self._draw()
            self._goal_probabilities[str(team_name)] = probabilities

        goals = np.atleast_1d(np.asarray(n))
        in_range = (goals >= 0) & (goals <= self.max_goals)
        return np.where(in_range, probabilities[np.clip(goals, 0, self.max_goals)], 0.0)

    def predict_outcome_proba(
        self, home_team: Sequence[str], away_team: Sequence[str]
    ) -> dict[str, np.ndarray]:
        return outcome_proba_from_scores(self, home_team, away_team, self.max_goals)
