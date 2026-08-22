"""
A team model whose predictions are random but well-formed.

Used as a control in `airsenal replay`: a season played on random scorelines is
the floor a real model has to clear.

It used to subclass `bpl.base.BaseMatchPredictor` while implementing only part
of the interface, so it inherited `predict_score_n_proba`, which reaches for a
`_teams_dict` this model never builds. Anything that asked it for a scoreline
probability - `fixture_probabilities`, and therefore `--team-model random` in
any real run - died with an AttributeError about a private bpl attribute.
It is now standalone and implements the whole of our own `TeamModel` protocol,
so the parts it does not support are the parts it does not claim.
"""

from collections.abc import Iterable
from typing import Any

import numpy as np

# Goals per team per match are modelled as a random distribution over 0..MAX_GOALS.
MAX_GOALS = 10


class RandomMatchPredictor:
    """Random, but valid, scoreline probabilities."""

    def __init__(self, max_goals: int = MAX_GOALS, random_state: int = 42) -> None:
        self.max_goals = max_goals
        self.teams: list[str] | None = None
        self.rng = np.random.default_rng(random_state)
        # per-team probability of scoring 0..max_goals, drawn once at fit time
        self._goal_probabilities: dict[str, np.ndarray] = {}

    def _draw(self) -> np.ndarray:
        """A random probability vector over 0..max_goals."""
        weights = self.rng.random(self.max_goals + 1)
        return weights / weights.sum()

    def fit(
        self, training_data: dict[str, Iterable[Any]], **kwargs: Any
    ) -> "RandomMatchPredictor":
        del kwargs
        home = training_data.get("home_team", [])
        away = training_data.get("away_team", [])
        self.teams = sorted({str(t) for t in [*home, *away]})
        if not self.teams:
            msg = "No teams found in training data."
            raise ValueError(msg)
        self._goal_probabilities = {team: self._draw() for team in self.teams}
        return self

    def add_new_team(self, team_name: str, **kwargs: Any) -> None:
        del kwargs
        if self.teams is None:
            self.teams = []
        if team_name in self.teams:
            return
        self.teams.append(team_name)
        self._goal_probabilities[team_name] = self._draw()

    def predict_score_n_proba(
        self,
        n: int | Iterable[int],
        team: str | Iterable[str],
        opponent: str | Iterable[str] = "",
        home: bool | None = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Probability that `team` scores each of `n` goals.

        The opponent and home advantage are ignored: this model is deliberately
        indifferent to who is playing.
        """
        del opponent, home, kwargs
        team_name = team if isinstance(team, str) else next(iter(team))
        probabilities = self._goal_probabilities.get(str(team_name))
        if probabilities is None:
            probabilities = self._draw()
            self._goal_probabilities[str(team_name)] = probabilities

        goals = np.atleast_1d(np.asarray(n))
        in_range = (goals >= 0) & (goals <= self.max_goals)
        return np.where(in_range, probabilities[np.clip(goals, 0, self.max_goals)], 0.0)
