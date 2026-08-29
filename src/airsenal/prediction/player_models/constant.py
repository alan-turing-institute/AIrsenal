"""A null player model: every player equally likely to score, assist, or neither."""

from dataclasses import dataclass

import numpy as np

from airsenal.prediction.protocols import PlayerFitData


@dataclass(frozen=True)
class ConstantPlayerConfig:
    """
    Settings for the null player model.

    The defaults are roughly the league-wide split of goal involvements, so the
    baseline is uninformative rather than obviously wrong.
    """

    prob_score: float = 0.25
    prob_assist: float = 0.2

    def __post_init__(self) -> None:
        if self.prob_score + self.prob_assist > 1:
            msg = (
                f"prob_score + prob_assist must not exceed 1, got "
                f"{self.prob_score} + {self.prob_assist}"
            )
            raise ValueError(msg)


class ConstantPlayerModel:
    """
    Every player equally likely to score, assist, or do neither.

    A null baseline, and a fast path when debugging something downstream of
    prediction, since it does no fitting at all.
    """

    def __init__(self, config: ConstantPlayerConfig | None = None) -> None:
        self.config = config or ConstantPlayerConfig()
        self.player_ids: np.ndarray | None = None

    def fit(self, data: PlayerFitData) -> "ConstantPlayerModel":
        self.player_ids = data["player_ids"]
        return self

    def _probabilities(self) -> np.ndarray:
        return np.array(
            [
                self.config.prob_score,
                self.config.prob_assist,
                1.0 - self.config.prob_score - self.config.prob_assist,
            ]
        )

    def get_probs(self) -> dict[str, np.ndarray]:
        if self.player_ids is None:
            msg = "Model has not been fitted yet."
            raise RuntimeError(msg)
        probs = self._probabilities()
        n = len(self.player_ids)
        return {
            "player_id": self.player_ids,
            "prob_score": np.full(n, probs[0]),
            "prob_assist": np.full(n, probs[1]),
            "prob_neither": np.full(n, probs[2]),
        }
