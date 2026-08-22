"""
A team model that gives every scoreline the same probability.

Useful in its own right as a null baseline for `airsenal replay` - if a real
team model does not beat "every scoreline is equally likely", it is not earning
its keep - and as a fast path when you are debugging something downstream of
prediction and do not want to wait for jax to fit anything.

It also means the registry indirection is exercised by shipped code rather than
only by test doubles.
"""

from collections.abc import Iterable
from typing import Any

import numpy as np

# Goals per team per match are modelled as uniform over 0..MAX_GOALS inclusive.
MAX_GOALS = 10


class ConstantTeamModel:
    """Every scoreline equally likely, whoever is playing."""

    def __init__(self, max_goals: int = MAX_GOALS) -> None:
        self.max_goals = max_goals
        self.teams: list[str] | None = None

    def fit(self, training_data: dict[str, Any], **kwargs: Any) -> "ConstantTeamModel":
        home = training_data.get("home_team", [])
        away = training_data.get("away_team", [])
        self.teams = sorted({str(t) for t in [*home, *away]})
        return self

    def add_new_team(self, team_name: str, **kwargs: Any) -> None:
        if self.teams is None:
            self.teams = []
        if team_name not in self.teams:
            self.teams.append(team_name)

    def predict_score_n_proba(
        self,
        n: int | Iterable[int],
        team: str | Iterable[str],  # noqa: ARG002
        opponent: str | Iterable[str],  # noqa: ARG002
        home: bool | None = True,  # noqa: ARG002
        **kwargs: Any,
    ) -> np.ndarray:
        goals = np.atleast_1d(np.asarray(n))
        # uniform over 0..max_goals, and zero for anything outside that
        probability = np.where(
            (goals >= 0) & (goals <= self.max_goals), 1.0 / (self.max_goals + 1), 0.0
        )
        return probability.astype(float)
