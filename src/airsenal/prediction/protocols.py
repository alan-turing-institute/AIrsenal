"""
What a prediction model has to provide.

Naming the shape means a signature does not change when the set of models does:
the three-way union of the bpl and random predictors used to be copy-pasted
across seven signatures, so adding a fourth team model meant finding all seven.

These are deliberately not `runtime_checkable`: `isinstance` against a Protocol
only checks that the method names exist, which is stringly-typed dispatch
wearing a type hint.
"""

from typing import Any, Protocol

import numpy as np


class PlayerModel(Protocol):
    """Predicts how a team's goals are shared out between its players."""

    def fit(self, data: dict[str, Any]) -> "PlayerModel":
        """
        Fit to the data, using the hyperparameters given at construction.

        Deliberately takes no `**kwargs`: it used to, and the numpyro model
        silently swallowed the hyperparameters the caller passed.
        """
        ...

    def get_probs(self) -> dict[str, np.ndarray]: ...


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
