"""
What a prediction model has to provide.

The three-way union of `ExtendedDixonColesMatchPredictor`,
`NeutralDixonColesMatchPredictor` and `RandomMatchPredictor`
was copy-pasted across seven signatures, so adding a fourth team model meant
finding all seven. Naming the shape instead means the signature does not change
when the set of models does.

These are deliberately not `runtime_checkable`: `isinstance` against a Protocol
only checks that the method names exist, which is the stringly-typed dispatch
this refactor is removing, wearing a type hint.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import numpy as np


class PlayerModel(Protocol):
    """Predicts how a team's goals are shared out between its players."""

    def fit(self, data: dict[str, Any]) -> PlayerModel:
        """
        Fit to the data, using the hyperparameters given at construction.

        Deliberately takes no `**kwargs`: it used to, and the numpyro model
        silently swallowed the hyperparameters the caller passed.
        """
        ...

    def get_probs(self) -> dict[str, np.ndarray]: ...


class TeamModel(Protocol):
    """Predicts match scorelines."""

    def fit(self, training_data: dict[str, Any], **kwargs: Any) -> TeamModel:
        """
        Fit to the data.

        `**kwargs` here is not ours to remove: bpl takes its hyperparameters at
        fit time, which is why the configs expose `fit_args()`.
        """
        ...

    def add_new_team(self, team_name: str, **kwargs: Any) -> None: ...

    def predict_score_n_proba(
        self, n: np.ndarray, team: str, opponent: str, home: bool = True, **kwargs: Any
    ) -> np.ndarray: ...
