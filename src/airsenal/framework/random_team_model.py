from collections.abc import Iterable

import numpy as np
from bpl.base import BaseMatchPredictor


class RandomMatchPredictor(BaseMatchPredictor):
    """A Random model for predicting match outcomes."""

    def __init__(self, num_samples: int = 1000, random_state: int = 42):
        self.teams: list[str] | None = None
        self.attack: np.ndarray | None = None
        self.defence: np.ndarray | None = None
        self.home_advantage: np.ndarray | None = None
        self.num_samples = num_samples
        self.rng = np.random.default_rng(random_state)

    def fit(self, training_data: dict[str, Iterable[str] | Iterable[float]], **kwargs):
        home_team = training_data["home_team"]
        away_team = training_data["away_team"]
        unique_teams = set(home_team) | set(away_team)
        self.teams = sorted([str(t) for t in unique_teams])
        if self.teams is None or len(self.teams) == 0:
            msg = "No teams found in training data."
            raise ValueError(msg)

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
        home_team = [home_team] if isinstance(home_team, str) else list(home_team)
        away_team = [away_team] if isinstance(away_team, str) else list(away_team)

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
        if self.teams is None:
            self.teams = []
        elif team_name in self.teams:
            msg = f"Team {team_name} already known to model."
            raise ValueError(msg)
        if self.attack is None:
            self.attack = np.empty((self.num_samples, 0))
        if self.defence is None:
            self.defence = np.empty((self.num_samples, 0))
        if self.home_advantage is None:
            self.home_advantage = np.empty((self.num_samples, 0))

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
        self.attack = np.concatenate((self.attack, attack[:, None]), axis=1)
        self.defence = np.concatenate((self.defence, defence[:, None]), axis=1)
        self.home_advantage = np.concatenate(
            (self.home_advantage, home_advantage[:, None]), axis=1
        )
