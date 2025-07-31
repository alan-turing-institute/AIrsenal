from collections.abc import Iterable

import jax.numpy as jnp
import numpy as np
from bpl.base import BaseMatchPredictor


class RandomMatchPredictor(BaseMatchPredictor):
    """A Random model for predicting match outcomes."""

    def __init__(self, num_samples=1000, random_state: int = 42):
        self.teams = None
        self.attack = None
        self.defence = None
        self.home_advantage = None
        self.num_samples = num_samples
        self.rng = np.random.default_rng(random_state)

    def fit(
        self,
        training_data: dict[str, Iterable[str] | Iterable[float]],
    ):
        home_team = training_data["home_team"]
        away_team = training_data["away_team"]
        self.teams = sorted(set(home_team) | set(away_team))

        self.attack = self.rng.normal(size=(self.num_samples, len(self.teams)))
        self.defence = self.rng.normal(size=(self.num_samples, len(self.teams)))
        self.home_advantage = self.rng.normal(size=(self.num_samples, len(self.teams)))
        self.corr_coef = self.rng.normal(size=(self.num_samples, len(self.teams)))
        self.rho = self.rng.normal(size=(self.num_samples,))
        return self

    def predict_score_proba(
        self,
        home_team: str | Iterable[str],
        away_team: str | Iterable[str],
        home_goals: int | Iterable[int],  # noqa: ARG002
        away_goals: int | Iterable[int],  # noqa: ARG002
    ) -> np.ndarray:
        home_team = [home_team] if isinstance(home_team, str) else home_team
        away_team = [away_team] if isinstance(away_team, str) else away_team

        home_probs = np.random.randn(self.num_samples, len(home_team))
        away_probs = np.random.randn(self.num_samples, len(away_team))

        sampled_probs = (
            np.random.randn(self.num_samples, len(home_team)) * home_probs * away_probs
        )
        return sampled_probs.mean(axis=0)

    def add_new_team(
        self,
        team_name: str,
        team_covariates: np.ndarray | None = None,  # noqa: ARG002
    ):
        if team_name in self.teams:
            msg = f"Team {team_name} already known to model."
            raise ValueError(msg)

        attack = np.random.randn(
            self.num_samples,
        )
        defence = np.random.randn(
            self.num_samples,
        )
        home_advantage = np.random.randn(
            self.num_samples,
        )

        self.teams.append(team_name)
        self.attack = jnp.concatenate((self.attack, attack[:, None]), axis=1)
        self.defence = jnp.concatenate((self.defence, defence[:, None]), axis=1)
        self.home_advantage = jnp.concatenate(
            (self.home_advantage, home_advantage[:, None]), axis=1
        )
