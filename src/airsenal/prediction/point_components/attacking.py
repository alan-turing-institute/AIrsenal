"""Points for goals and assists."""

import pandas as pd
from scipy.stats import multinomial
from sqlalchemy.orm import Session

from airsenal.game.scoring import points_for_assist, points_for_goal
from airsenal.prediction.protocols import ComponentRequest


def get_attacking_points(
    position: str,
    minutes: int | float,
    team_score_prob: dict[int, float],
    player_prob: pd.Series | dict[str, float],
) -> float:
    """Calculate expected attacking points (goals and assists) for a player."""
    if minutes == 0.0:
        return 0.0

    pr_score = (minutes / 90.0) * player_prob["prob_score"]
    pr_assist = (minutes / 90.0) * player_prob["prob_assist"]
    pr_neither = 1.0 - pr_score - pr_assist
    multinom_probs = (pr_score, pr_assist, pr_neither)

    def _get_partitions(n: int) -> list[list[int]]:
        partitions = []
        for i in range(n + 1):
            for j in range(n - i + 1):
                partitions.append([i, j, n - i - j])
        return partitions

    def _get_partition_score(partition: list[int]) -> int:
        return (
            points_for_goal[position] * partition[0] + points_for_assist * partition[1]
        )

    exp_points = 0.0
    for ngoals, score_n_prob in team_score_prob.items():
        if ngoals > 0:
            partitions = _get_partitions(ngoals)
            probabilities = multinomial.pmf(
                partitions, n=[ngoals] * len(partitions), p=multinom_probs
            )
            scores = map(_get_partition_score, partitions)
            exp_score_inner = sum(
                pi * si for pi, si in zip(probabilities, scores, strict=True)
            )
            exp_points += exp_score_inner * score_n_prob
    return exp_points


class AttackingComponent:
    """
    The player's share of however many goals their team scores.

    Their share comes from the player model and the number of goals from the
    team model, so this is the one component that needs both - which is why the
    team's goal distribution is in every `ComponentRequest`.
    """

    name = "attacking"

    def fit(
        self, gameweek: int, season: str, dbsession: Session
    ) -> "AttackingComponent":
        """Nothing to fit: the player and team models have been fitted already."""
        del gameweek, season, dbsession
        return self

    def expected_points(self, request: ComponentRequest) -> float:
        return get_attacking_points(
            request.position,
            request.minutes,
            request.team_score_probability,
            {
                "prob_score": request.prob_score,
                "prob_assist": request.prob_assist,
            },
        )
