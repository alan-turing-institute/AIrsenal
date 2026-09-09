"""A player model fitted to expected goals and assists."""

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from airsenal.core.logging import get_logger
from airsenal.game.enums import Position
from airsenal.prediction.player_models.conjugate import ConjugatePlayerModel
from airsenal.prediction.player_models.scaling import (
    FloatArray,
    scale_goals_by_minutes,
)
from airsenal.prediction.protocols import (
    PlayerFitData,
    PlayerInvolvement,
    no_expected_goals_message,
)

logger = get_logger(__name__)

# Weight a match at exp(-epsilon * years ago). Swept with
# tools/tune_player_time_weighting.py over 2324, 2425 and 2526 but has almost no impact
DEFAULT_XG_PLAYER_EPSILON = 0.2

# How hard a player is pulled towards his position's mean. Swept per position and chosen
# held out; see docs/xg-models.md.
DEFAULT_XG_N_GOALS_PRIOR: Mapping[str, int] = {
    str(Position.GK): 700,
    str(Position.DEF): 15,
    str(Position.MID): 6,
    str(Position.FWD): 60,
}

# How much of the fitting target is actual goals scored rather than xG. A
# small weight on them beats none in every season, at every position (but does not help
# the team model)
DEFAULT_XG_GOAL_WEIGHT = 0.15


@dataclass(frozen=True)
class XGPlayerConfig:
    """
    How a player's share of their team's goals is fitted.

    Args:
        epsilon: Time-weighting decay, per year, or None for no weighting at all.
            A match a season old counts `exp(-epsilon)` of one played yesterday.
        n_goals_prior: How many goals' worth of the pooled average a player is
            credited with before their own are counted, which is what sets how
            hard a thin record is pulled towards the pool. One number for every
            position, or one per position - which is what the default is,
            because the four want very different amounts.
        rescale_weights: Rescale each player's time weights to sum to the number
            of matches they count for.
        calibrate: Scale expected goals and assists by what the same window's
            players actually converted them into. Expected goals need almost
            none of this - the league scores what it is expected to - but FPL
            awards assists that no chance-creating pass is credited for, so
            uncalibrated expected assists undershoot by about 30%.
        goal_weight: How much of the fitting target is the real involvement
            rather than the expected one. Zero fits to expected goals alone, one
            makes this the conjugate model with extra steps.
    """

    epsilon: float | None = DEFAULT_XG_PLAYER_EPSILON
    n_goals_prior: int | Mapping[str, int] = field(
        default_factory=lambda: dict(DEFAULT_XG_N_GOALS_PRIOR)
    )
    rescale_weights: bool = True
    calibrate: bool = True
    goal_weight: float = DEFAULT_XG_GOAL_WEIGHT

    def __post_init__(self) -> None:
        if not 0.0 <= self.goal_weight <= 1.0:
            msg = (
                "goal_weight is a mixing weight, so it lives in [0, 1]: "
                f"{self.goal_weight}"
            )
            raise ValueError(msg)

    def prior_for(self, position: str | None) -> int:
        """
        The shrinkage to fit this position's players with.

        A per-position prior needs the training data to say which position it
        is about, and `PlayerFitData` only carries that when it came from
        `process_player_data` - so a caller who assembled their own is told
        what is missing rather than quietly given somebody else's shrinkage.
        """
        if isinstance(self.n_goals_prior, int):
            return self.n_goals_prior
        if position is None:
            msg = (
                "n_goals_prior is one value per position, and this training "
                "data does not say which position it is about. Pass a single "
                "number instead, or include `position` in the fit data as "
                "`process_player_data` does."
            )
            raise ValueError(msg)
        if position not in self.n_goals_prior:
            msg = (
                f"No n_goals_prior for position {position!r}, only "
                f"{sorted(self.n_goals_prior)}"
            )
            raise ValueError(msg)
        return self.n_goals_prior[position]


class XGPlayerModel:
    """
    Who takes a share of a team's goals, read from expected goals and assists.

    The same conjugate Dirichlet update as `ConjugatePlayerModel` - and its
    prior, its minutes scaling and its time weighting - over a different count.
    Where that model asks what fraction of its team's goals a player scored,
    this one asks what fraction of its expected goals the player was expected to
    score.

    Not purely, though: `goal_weight` mixes a small fraction of what the player
    actually did back into the target - `DEFAULT_XG_GOAL_WEIGHT` of it - because
    at player level, unlike team level, that measures better than either alone.

    **Every match counts.** A goalless match tells the goals model nothing
    about who its team's scorers are, and `scale_goals_by_minutes` drops it. A
    goalless match still has expected goals in it, so it is still evidence.
    A fifth to a quarter of team-matches are goalless.
    """

    def __init__(self, config: XGPlayerConfig | None = None):
        self.config = config or XGPlayerConfig()
        self.player_ids: FloatArray | None = None
        self.prior: FloatArray | None = None
        self.posterior: FloatArray | None = None
        self.mean_probabilities: FloatArray | None = None
        # What a unit of expected goals, and one of expected assists, was worth
        # in the window fitted - 1.0 each when `calibrate` is off.
        self.finishing = 1.0
        self.creation = 1.0

    @property
    def epsilon(self) -> float | None:
        return self.config.epsilon

    def fit(self, data: PlayerFitData) -> "XGPlayerModel":
        n_goals_prior = self.config.prior_for(data.get("position"))
        logger.info(
            "Fitting XGPlayerModel for %s with epsilon=%s, n_goals_prior=%s, "
            "calibrate=%s, goal_weight=%s",
            data.get("position", "an unnamed position"),
            self.config.epsilon,
            n_goals_prior,
            self.config.calibrate,
            self.config.goal_weight,
        )
        self.player_ids = data["player_ids"]
        scaled = scale_goals_by_minutes(
            goals=self._involvement_counts(data),
            minutes=data["minutes"],
            time_diff=data.get("time_diff"),
            epsilon=self.config.epsilon,
            rescale_weights=self.config.rescale_weights,
        )
        self.prior = ConjugatePlayerModel.get_prior(scaled, n_goals_prior=n_goals_prior)
        self.posterior = ConjugatePlayerModel.get_posterior(self.prior, scaled)
        self.mean_probabilities = self.posterior / self.posterior.sum(axis=1)[:, None]
        return self

    def _involvement_counts(self, data: PlayerFitData) -> FloatArray:
        """
        What this model is fitted to: (scoring, assisting, neither), per match.

        The same shape and meaning as `PlayerFitData["y"]`, in expected goals
        rather than goals, so `scale_goals_by_minutes` handles it unchanged. The
        three sum to the team's expected goals in the match, which is what makes
        the fitted numbers a share of one goal; only that total is read
        downstream, so the third entry is left exact rather than clipped where a
        player was expected to be involved in more than his team was.

        A match with no expected goals recorded is zeroed, which is how
        `scale_goals_by_minutes` is already told a match carries nothing. The
        same test excludes a padding row, whose team expected goals are zero -
        see `features.blank_player_row`.
        Records the calibration it applied, which is part of what the fit found.
        """
        missing = [
            key
            for key in ("expected_goals", "expected_assists", "team_expected_goals")
            if key not in data
        ]
        if missing:
            msg = (
                "XGPlayerModel is fitted to expected goals, and the training "
                f"data is missing {', '.join(missing)}. `process_player_data` "
                "provides them."
            )
            raise ValueError(msg)
        expected_goals = np.asarray(data["expected_goals"], dtype=float)
        expected_assists = np.asarray(data["expected_assists"], dtype=float)
        team_expected = np.asarray(data["team_expected_goals"], dtype=float)
        goals = np.asarray(data["y"], dtype=float)

        recorded = (
            np.isfinite(expected_goals)
            & np.isfinite(expected_assists)
            & np.isfinite(team_expected)
            & (team_expected > 0)
        )
        if not recorded.any():
            msg = no_expected_goals_message(
                type(self).__name__, "--player-model conjugate"
            )
            raise ValueError(msg)

        self.finishing, self.creation = self._calibration(data, recorded)
        scoring = self.finishing * np.where(recorded, expected_goals, 0.0)
        assisting = self.creation * np.where(recorded, expected_assists, 0.0)
        total = np.where(recorded, team_expected, 0.0)
        counts = np.stack([scoring, assisting, total - scoring - assisting], axis=2)
        if self.config.goal_weight:
            # Mixed per match, and only over the matches this model can see, so
            # the weight is what it says it is rather than also standing in
            # wherever the expected goals are missing.
            counts = (1.0 - self.config.goal_weight) * counts + (
                self.config.goal_weight * np.where(recorded[..., None], goals, 0.0)
            )
        return counts

    def _calibration(
        self, data: PlayerFitData, recorded: np.ndarray
    ) -> tuple[float, float]:
        """What a unit of expected goals, and one of expected assists, was worth."""
        if not self.config.calibrate:
            return 1.0, 1.0
        goals = np.asarray(data["y"], dtype=float)

        def ratio(actual: np.ndarray, expected: np.ndarray) -> float:
            total = float(np.asarray(expected, dtype=float)[recorded].sum())
            return float(actual[recorded].sum() / total) if total > 0 else 1.0

        return (
            ratio(goals[..., 0], data["expected_goals"]),
            ratio(goals[..., 1], data["expected_assists"]),
        )

    def predict_involvement(self) -> PlayerInvolvement:
        if self.player_ids is None or self.mean_probabilities is None:
            msg = "The xG player model has not been fitted yet."
            raise RuntimeError(msg)
        return PlayerInvolvement(
            player_ids=self.player_ids,
            prob_score=self.mean_probabilities[:, 0],
            prob_assist=self.mean_probabilities[:, 1],
            prob_neither=self.mean_probabilities[:, 2],
        )
