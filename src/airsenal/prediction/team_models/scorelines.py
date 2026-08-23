"""
Deriving match-outcome probabilities from per-team scoreline probabilities.

The null models predict each team's goals independently, so win/draw/loss falls
out of convolving the two distributions. bpl does this itself, so
`DixonColesTeamModel` does not come through here.
"""

from collections.abc import Sequence

import numpy as np

from airsenal.core.scoring import MAX_GOALS
from airsenal.prediction.protocols import TeamModel


def outcome_proba_from_scores(
    model: TeamModel,
    home_team: Sequence[str],
    away_team: Sequence[str],
    max_goals: int = MAX_GOALS,
) -> dict[str, np.ndarray]:
    """Win/draw/loss probabilities per fixture, from independent goal counts."""
    goals = np.arange(max_goals + 1)
    home_win, draw, away_win = [], [], []
    for home, away in zip(home_team, away_team, strict=True):
        # outer[h, a] = P(home scores h) * P(away scores a)
        outer = np.outer(
            model.predict_score_n_proba(goals, home, away, home=True),
            model.predict_score_n_proba(goals, away, home, home=False),
        )
        home_win.append(float(np.tril(outer, -1).sum()))
        draw.append(float(np.trace(outer)))
        away_win.append(float(np.triu(outer, 1).sum()))
    return {
        "home_win": np.array(home_win),
        "draw": np.array(draw),
        "away_win": np.array(away_win),
    }
