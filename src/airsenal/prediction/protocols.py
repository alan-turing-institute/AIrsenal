"""What a prediction model has to provide."""

from collections.abc import Sequence
from typing import Any, NotRequired, Protocol, TypedDict

import numpy as np


class PlayerFitData(TypedDict):
    """
    Everything a player model is fitted to, from `features.process_player_data`.

    One position's players, over the matches in the fitting window. Every player
    has a row for every match, zero-padded where they did not appear, so `y` and
    `minutes` are rectangular and `nplayer`/`nmatch` are their dimensions.
    """

    # (n_players,) the players these rows are about, sorted
    player_ids: np.ndarray
    nplayer: int
    nmatch: int
    # (n_players, n_matches) minutes played
    minutes: np.ndarray
    # (n_players, n_matches, 3) goal involvements, the last axis being
    # (goals, assists, neither). The three sum to the goals the player's team
    # scored in that match.
    y: np.ndarray
    # (3,) Dirichlet prior concentrations over that same last axis. Strictly
    # positive, or a model that builds a real Dirichlet from it cannot be fitted.
    alpha: np.ndarray
    # (n_players, n_matches) years between the match and the gameweek being
    # predicted, for a model that weights recent matches more heavily
    time_diff: np.ndarray


class TeamFitData(TypedDict):
    """
    Everything a team model is fitted to, from `team_models.get_training_data`.

    One entry per past match, so every array here has the same length.
    """

    home_team: np.ndarray
    away_team: np.ndarray
    home_goals: np.ndarray
    away_goals: np.ndarray
    # years between the match and the gameweek being predicted
    time_diff: np.ndarray
    neutral_venue: np.ndarray
    game_weights: np.ndarray
    # FIFA ratings per team name, absent when fitting without them. A promoted
    # team has no results, so its ratings are what `add_new_team` stands in with.
    team_covariates: NotRequired[dict[str, np.ndarray]]


class PlayerModel(Protocol):
    """Predicts how a team's goals are shared out between its players."""

    def fit(self, data: PlayerFitData) -> "PlayerModel":
        """
        Fit to the data, using the hyperparameters given at construction.

        Deliberately takes no `**kwargs`, so a hyperparameter a model does not
        implement is an error rather than something it silently swallows.
        """
        ...

    def get_probs(self) -> dict[str, np.ndarray]:
        """
        Per-player probabilities of scoring, assisting, or neither, for a goal.

        Keys: "player_id", "prob_score", "prob_assist", "prob_neither", each an
        array of shape (n_players,). The last three sum to one per player.
        """
        ...


class TeamModel(Protocol):
    """Predicts match scorelines."""

    @property
    def teams(self) -> list[str] | None:
        """The teams this model knows about, or None before it is fitted."""
        ...

    def fit(self, training_data: TeamFitData) -> "TeamModel":
        """
        Fit to the data, using the settings given at construction.

        Like `PlayerModel.fit`, this takes no `**kwargs`. bpl wants its
        time-weighting arguments at fit time, so `DixonColesTeamModel` holds
        them and passes them on itself.
        """
        ...

    def add_new_team(self, team_name: str, **kwargs: Any) -> None:
        """
        Teach the model a team that has no results to fit to, such as a promoted one.

        Called after `fit`, once per unknown team. `team_covariates` is passed as
        a keyword when the model is being fitted with FIFA ratings; a model that
        does not use covariates ignores it.
        """
        ...

    def predict_score_n_proba(
        self, n: np.ndarray, team: str, opponent: str, home: bool = True, **kwargs: Any
    ) -> np.ndarray:
        """
        The probability of `team` scoring each goal count in `n` against `opponent`.

        Returns an array the same length as `n`. `home` says which side of the
        fixture `team` is on.
        """
        ...

    def predict_outcome_proba(
        self, home_team: Sequence[str], away_team: Sequence[str]
    ) -> dict[str, np.ndarray]:
        """
        Win, draw and loss probabilities for each fixture.

        Keyed "home_win", "draw" and "away_win". Here rather than fetched with
        `getattr` at the one call site: a model that cannot answer should fail to
        type-check, not fail at run time.
        """
        ...
