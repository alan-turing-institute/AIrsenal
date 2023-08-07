from typing import Dict, Iterable, Optional, Union

import jax.numpy as jnp
import numpy as np
from bpl.base import BaseMatchPredictor


class RandomMatchPredictor(BaseMatchPredictor):
    """A Random model for predicting match outcomes."""

    def __init__(self, num_samples=1000):
        self.teams = None
        self.attack = None
        self.defence = None
        self.home_advantage = None
        self.num_samples = num_samples

    def fit(
        self,
        training_data: Dict[str, Union[Iterable[str], Iterable[float]]],
        random_state: int = 42,
    ):
        home_team = training_data["home_team"]
        away_team = training_data["away_team"]
        self.teams = sorted(list(set(home_team) | set(away_team)))

        self.attack = np.random.randn(self.num_samples, len(self.teams))
        self.defence = np.random.randn(self.num_samples, len(self.teams))
        self.home_advantage = np.random.randn(self.num_samples, len(self.teams))
        self.corr_coef = np.random.randn(
            self.num_samples,
        )
        self.rho = np.random.randn(
            self.num_samples,
        )
        return self

    def predict_score_proba(
        self,
        home_team: Union[str, Iterable[str]],
        away_team: Union[str, Iterable[str]],
        home_goals: Union[int, Iterable[int]],
        away_goals: Union[int, Iterable[int]],
    ) -> np.array:
        home_team = [home_team] if isinstance(home_team, str) else home_team
        away_team = [away_team] if isinstance(away_team, str) else away_team

        home_probs = np.random.randn(self.num_samples, len(home_team))
        away_probs = np.random.randn(self.num_samples, len(away_team))

        sampled_probs = (
            np.random.randn(self.num_samples, len(home_team)) * home_probs * away_probs
        )
        return sampled_probs.mean(axis=0)

    def add_new_team(self, team_name: str, team_covariates: Optional[np.array] = None):
        if team_name in self.teams:
            raise ValueError("Team {} already known to model.".format(team_name))

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
