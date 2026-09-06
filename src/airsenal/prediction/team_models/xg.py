"""A team model fitted to expected goals rather than to goals."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from airsenal.prediction.protocols import TeamFitData


@dataclass(frozen=True)
class XGTeamConfig:
    """
    How the attack and defence ratings are fitted.

    Args:
        n_iterations: Passes of the alternating fit. Each pass re-rates every
            attack against the current defences and vice versa, which is what
            separates a good attack from an easy schedule. It converges quickly;
            more than a handful buys nothing.
        prior_matches: How many matches of exactly average performance every
            team is credited with before its own are counted. A team with three
            matches played is mostly the league average; one with thirty is
            mostly itself.
    """

    n_iterations: int = 10
    prior_matches: float = 5.0


class XGTeamModel:
    """
    Expected goals from a team's attack, its opponent's defence, and the venue.

    Fitted to the expected goals in past matches rather than to the goals
    themselves, on the usual argument that a shot's chance of going in says more
    about the next match than whether it happened to. It predicts a mean and
    nothing else - there is no distribution over goal *counts* to be had from a
    continuous quantity - so it is an `ExpectedGoalsTeamModel`, and reaches the
    points calculation through `PoissonScorelines`.

    The ratings are multiplicative and average one: an attack of 1.2 creates a
    fifth more than the league does, against the same defence at the same venue.
    """

    def __init__(self, config: XGTeamConfig | None = None):
        self.config = config or XGTeamConfig()
        self.teams: list[str] | None = None
        self.attack: dict[str, float] = {}
        self.defence: dict[str, float] = {}
        # The league's expected goals per match at each venue, which is where
        # home advantage lives: it is the same for every team.
        self.home_mean = 0.0
        self.away_mean = 0.0

    def fit(self, training_data: TeamFitData) -> "XGTeamModel":
        home_team = np.asarray(training_data["home_team"])
        away_team = np.asarray(training_data["away_team"])
        if (
            "home_expected_goals" not in training_data
            or "away_expected_goals" not in training_data
        ):
            msg = (
                "XGTeamModel is fitted to expected goals, which this training "
                "data does not carry. `get_training_data` provides them."
            )
            raise ValueError(msg)
        home_xg = np.asarray(training_data["home_expected_goals"], dtype=float)
        away_xg = np.asarray(training_data["away_expected_goals"], dtype=float)

        played = ~(np.isnan(home_xg) | np.isnan(away_xg))
        if not played.any():
            msg = "No match in the training data has expected goals recorded"
            raise ValueError(msg)
        home_team, away_team = home_team[played], away_team[played]
        home_xg, away_xg = home_xg[played], away_xg[played]

        self.teams = sorted({*home_team.tolist(), *away_team.tolist()})
        self.home_mean = float(home_xg.mean())
        self.away_mean = float(away_xg.mean())
        self.attack = dict.fromkeys(self.teams, 1.0)
        self.defence = dict.fromkeys(self.teams, 1.0)

        # Alternating multiplicative fit: rate each attack against the defences
        # it actually faced, then each defence against the attacks it faced.
        prior = self.config.prior_matches * (self.home_mean + self.away_mean) / 2
        for _ in range(self.config.n_iterations):
            self.attack = self._rate(
                home_team,
                away_team,
                home_xg,
                away_xg,
                self.defence,
                prior,
                scoring=True,
            )
            self.defence = self._rate(
                home_team,
                away_team,
                home_xg,
                away_xg,
                self.attack,
                prior,
                scoring=False,
            )
        return self

    def _rate(
        self,
        home_team: np.ndarray,
        away_team: np.ndarray,
        home_xg: np.ndarray,
        away_xg: np.ndarray,
        against: dict[str, float],
        prior: float,
        *,
        scoring: bool,
    ) -> dict[str, float]:
        """
        One side of the alternating fit, for every team at once.

        `scoring` rates attacks against the opponents' defences; otherwise it
        rates defences against the opponents' attacks. Either way a team's
        rating is what it managed over what an average team would have managed
        in the same matches.
        """
        assert self.teams is not None
        actual = dict.fromkeys(self.teams, prior)
        expected = dict.fromkeys(self.teams, prior)
        for home, away, for_home, for_away in zip(
            home_team, away_team, home_xg, away_xg, strict=True
        ):
            # (the team being rated, its opponent, what happened, the venue mean)
            sides = (
                (
                    (home, away, for_home, self.home_mean),
                    (away, home, for_away, self.away_mean),
                )
                if scoring
                else (
                    (away, home, for_home, self.home_mean),
                    (home, away, for_away, self.away_mean),
                )
            )
            for team, opponent, xg, venue_mean in sides:
                actual[team] += float(xg)
                expected[team] += venue_mean * against.get(opponent, 1.0)
        rated = {
            team: actual[team] / expected[team] if expected[team] else 1.0
            for team in self.teams
        }
        # The ratings and the venue means are only identified up to a constant,
        # so pin the ratings at an average of one and leave the level to them.
        mean = float(np.mean(list(rated.values())))
        return (
            {team: rating / mean for team, rating in rated.items()} if mean else rated
        )

    def add_new_team(self, team_name: str, **kwargs: Any) -> None:
        """A team with no matches is the league average until it plays some."""
        del kwargs
        if self.teams is None:
            self.teams = []
        if team_name not in self.teams:
            self.teams.append(team_name)
            self.teams.sort()
        self.attack.setdefault(team_name, 1.0)
        self.defence.setdefault(team_name, 1.0)

    def predict_expected_goals(
        self, team: str, opponent: str, home: bool = True, **kwargs: Any
    ) -> float:
        del kwargs
        if self.teams is None:
            msg = "The xG team model has not been fitted yet."
            raise RuntimeError(msg)
        venue_mean = self.home_mean if home else self.away_mean
        return venue_mean * self.attack.get(team, 1.0) * self.defence.get(opponent, 1.0)
