"""A team model fitted to expected goals rather than to goals."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from airsenal.prediction.protocols import TeamFitData, no_expected_goals_message

# Weight a match at exp(-epsilon * years ago). Swept over 2425 and 2526 with
# tools/tune_team_time_weighting.py but only worth about a thousandth of a nat over no
# weighting at all.
DEFAULT_XG_EPSILON = 0.6


@dataclass(frozen=True)
class XGTeamConfig:
    """
    How the attack and defence ratings are fitted.

    Args:
        max_iterations: Most passes of the alternating fit to make. Each pass
            re-rates every attack against the current defences and vice versa. It
            stops as soon as the ratings stop moving by `tolerance`, which
            on a real schedule takes ~10 iterations.
        tolerance: How still the ratings have to be to call the fit done - the
            largest change in any rating over a pass.
        prior_matches: How many matches of exactly average performance every
            team is credited with before its own are counted.
        epsilon: Time-weighting decay, per year. A match a season old counts
            `exp(-epsilon)` of one played yesterday. Zero weights every match in
            the window equally.
        promoted_like_bottom: How many of the worst teams a side with no record
            is assumed to resemble, or None to assume it is an average one.
            Evaluated not to be beneficial, so it is off by default. See
            docs/xg-models.md for the numbers.
    """

    max_iterations: int = 100
    tolerance: float = 1e-12
    prior_matches: float = 5.0
    epsilon: float = DEFAULT_XG_EPSILON
    promoted_like_bottom: int | None = None


class XGTeamModel:
    """
    Expected goals from a team's attack, its opponent's defence, and the venue.

    Fitted to the expected goals in past matches rather than to the goals
    themselves. It predicts a mean and nothing else, so it is an
    `ExpectedGoalsTeamModel`, and its `TEAM_MODELS` entry wraps it in
    `ConwayMaxwellScorelines`.

    The ratings are multiplicative and average one: an attack of 1.2 creates a
    fifth more than the league does, against the same defence at the same venue.
    """

    def __init__(self, config: XGTeamConfig | None = None):
        self.config = config or XGTeamConfig()
        self.teams: list[str] | None = None
        self.attack: dict[str, float] = {}
        self.defence: dict[str, float] = {}
        # The league's expected goals per match at each venue, the same for every team.
        self.home_mean = 0.0
        self.away_mean = 0.0
        # What a new team with no data is assumed to be
        self.promoted_attack = 1.0
        self.promoted_defence = 1.0

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
            msg = no_expected_goals_message(
                type(self).__name__, "--team-model extended"
            )
            raise ValueError(msg)
        home_team, away_team = home_team[played], away_team[played]
        home_xg, away_xg = home_xg[played], away_xg[played]

        weights = self._weights(training_data, played)
        self.teams = sorted({*home_team.tolist(), *away_team.tolist()})
        self.home_mean = float(np.average(home_xg, weights=weights))
        self.away_mean = float(np.average(away_xg, weights=weights))
        self.attack = dict.fromkeys(self.teams, 1.0)
        self.defence = dict.fromkeys(self.teams, 1.0)

        # Alternating multiplicative fit: rate each attack against the defences
        # it actually faced, then each defence against the attacks it faced.
        prior = self.config.prior_matches * (self.home_mean + self.away_mean) / 2
        for _ in range(self.config.max_iterations):
            before = (self.attack, self.defence)
            self.attack = self._rate(
                home_team,
                away_team,
                home_xg,
                away_xg,
                weights,
                self.defence,
                prior,
                scoring=True,
            )
            self.defence = self._rate(
                home_team,
                away_team,
                home_xg,
                away_xg,
                weights,
                self.attack,
                prior,
                scoring=False,
            )
            if self._settled(before):
                break
        self._rate_a_promoted_team()
        return self

    def _settled(self, before: tuple[dict[str, float], dict[str, float]]) -> bool:
        """Whether the last pass left every rating unchanged."""
        was_attack, was_defence = before
        return all(
            abs(now[team] - was[team]) < self.config.tolerance
            for now, was in ((self.attack, was_attack), (self.defence, was_defence))
            for team in now
        )

    def _weights(self, training_data: TeamFitData, played: np.ndarray) -> np.ndarray:
        """How much each match counts, rescaled to sum to the number of matches."""
        if not self.config.epsilon:
            return np.ones(int(played.sum()))
        time_diff = np.asarray(training_data["time_diff"], dtype=float)[played]
        weights = np.asarray(np.exp(-self.config.epsilon * time_diff), dtype=float)
        total = float(weights.sum())
        return weights * (len(weights) / total) if total else weights

    def _rate_a_promoted_team(self) -> None:
        """
        What to assume about a team with no record in the window.

        The league average by default, which is what an unrated team gets from
        ratings that average one. `promoted_like_bottom` instead assumes it
        resembles the worst teams that do have a record.
        """
        assert self.teams is not None
        if not self.teams or self.config.promoted_like_bottom is None:
            return
        # creating more and conceding less is better - so the worst teams have the
        # smallest ratios.
        worst = sorted(
            self.teams, key=lambda team: self.attack[team] / self.defence[team]
        )[: max(1, self.config.promoted_like_bottom)]
        self.promoted_attack = float(np.mean([self.attack[t] for t in worst]))
        self.promoted_defence = float(np.mean([self.defence[t] for t in worst]))

    def _rate(
        self,
        home_team: np.ndarray,
        away_team: np.ndarray,
        home_xg: np.ndarray,
        away_xg: np.ndarray,
        weights: np.ndarray,
        against: dict[str, float],
        prior: float,
        *,
        scoring: bool,
    ) -> dict[str, float]:
        """
        One side of the alternating fit, for every team at once.

        `scoring` rates attacks against the opponents' defences; otherwise it
        rates defences against the opponents' attacks.
        """
        assert self.teams is not None
        actual = dict.fromkeys(self.teams, prior)
        expected = dict.fromkeys(self.teams, prior)
        for home, away, for_home, for_away, weight in zip(
            home_team, away_team, home_xg, away_xg, weights, strict=True
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
                actual[team] += float(weight) * float(xg)
                expected[team] += (
                    float(weight) * venue_mean * against.get(opponent, 1.0)
                )
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
        if self.teams is None:
            self.teams = []
        if team_name not in self.teams:
            self.teams.append(team_name)
            self.teams.sort()
        self.attack.setdefault(team_name, self.promoted_attack)
        self.defence.setdefault(team_name, self.promoted_defence)

    def predict_expected_goals(
        self, team: str, opponent: str, home: bool = True, **kwargs: Any
    ) -> float:
        if self.teams is None:
            msg = "The xG team model has not been fitted yet."
            raise RuntimeError(msg)
        venue_mean = self.home_mean if home else self.away_mean
        return (
            venue_mean
            * self.attack.get(team, self.promoted_attack)
            * self.defence.get(opponent, self.promoted_defence)
        )
