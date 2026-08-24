"""
A team model that gives every scoreline the same probability.

A null baseline for `airsenal replay`: if a real team model does not beat
"every scoreline is equally likely", it is not earning its keep. Also a fast
path when debugging something downstream of prediction, since fitting it costs
nothing.
"""

from collections.abc import Iterable
from typing import Any

import numpy as np

from airsenal.core.registry import ConfigError

# Goals per team per match are modelled as uniform over 0..MAX_GOALS inclusive.
MAX_GOALS = 10


class ConstantTeamModel:
    """Every scoreline equally likely, whoever is playing."""

    def __init__(
        self, max_goals: int = MAX_GOALS, *, epsilon: float | None = None
    ) -> None:
        if epsilon is not None:
            msg = "the constant team model has no time weighting, so no epsilon"
            raise ConfigError(msg)
        self.max_goals = max_goals
        self.teams: list[str] | None = None

    def fit(self, training_data: dict[str, Any]) -> "ConstantTeamModel":
        home = training_data.get("home_team", [])
        away = training_data.get("away_team", [])
        self.teams = sorted({str(t) for t in [*home, *away]})
        return self

    def add_new_team(self, team_name: str, **kwargs: Any) -> None:
        del kwargs
        if self.teams is None:
            self.teams = []
        if team_name not in self.teams:
            self.teams.append(team_name)

    def predict_score_n_proba(
        self,
        n: int | Iterable[int],
        team: str | Iterable[str],
        opponent: str | Iterable[str] = "",
        home: bool | None = True,
        **kwargs: Any,
    ) -> np.ndarray:
        del team, opponent, home, kwargs
        goals = np.atleast_1d(np.asarray(n))
        # uniform over 0..max_goals, and zero for anything outside that
        probability = np.where(
            (goals >= 0) & (goals <= self.max_goals), 1.0 / (self.max_goals + 1), 0.0
        )
        return probability.astype(float)
