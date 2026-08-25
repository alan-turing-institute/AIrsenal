"""
What a prediction model has to provide.

Naming the shape means a signature does not change when the set of models does.

These are deliberately not `runtime_checkable`, so nothing dispatches on
`isinstance` against them.
"""

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np


class PlayerModel(Protocol):
    """Predicts how a team's goals are shared out between its players."""

    def fit(self, data: dict[str, Any]) -> "PlayerModel":
        """
        Fit to the data, using the hyperparameters given at construction.

        Deliberately takes no `**kwargs`, so a hyperparameter a model does not
        implement is an error rather than something it silently swallows.

        `data` must have at least these keys:
        - "y": (n_players, n_matches, 3) goal involvements per match, the last
          axis being (goals, assists, neither)
        - "player_ids": (n_players,)
        - "minutes": (n_players, n_matches) minutes played
        """
        ...

    def get_probs(self) -> dict[str, np.ndarray]:
        """
        Per-player probabilities of scoring, assisting, or neither, for a goal.

        Keys: "player_id", "prob_score", "prob_assist", "prob_neither", each an
        array of shape (n_players,).
        """
        ...


class TeamModel(Protocol):
    """Predicts match scorelines."""

    @property
    def teams(self) -> list[str] | None:
        """The teams this model knows about, or None before it is fitted."""
        ...

    def fit(self, training_data: dict[str, Any]) -> "TeamModel":
        """
        Fit to the data, using the settings given at construction.

        Like `PlayerModel.fit`, this takes no `**kwargs`. bpl wants its
        time-weighting arguments at fit time, so `DixonColesTeamModel` holds
        them and passes them on itself.
        """
        ...

    def add_new_team(self, team_name: str, **kwargs: Any) -> None: ...

    def predict_score_n_proba(
        self, n: np.ndarray, team: str, opponent: str, home: bool = True, **kwargs: Any
    ) -> np.ndarray: ...

    def predict_outcome_proba(
        self, home_team: Sequence[str], away_team: Sequence[str]
    ) -> dict[str, np.ndarray]:
        """
        Win/draw/loss probabilities for each fixture, keyed "home_win"/"draw"/
        "away_win".

        Here rather than fetched with `getattr` at the one call site: a model that
        cannot answer should fail to type-check, not fail at run time.
        """
        ...
