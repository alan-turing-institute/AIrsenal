"""Shrinking a player's own history towards the average for their position."""

import pandas as pd

from airsenal.game.scoring import MIN_MINUTES_FULL, MIN_MINUTES_SHORT


def mean_group_prior(
    df: pd.DataFrame,
    group_col: str,
    mean_col: str,
    n_prior: int = 10,
    prior_by_position: bool = False,
) -> pd.Series:
    """Compute empirical Bayes group means with a prior weight."""
    groups = df.groupby(group_col)[mean_col].agg(["count", "sum"])
    group_counts = groups["count"]
    group_sums = groups["sum"]

    if prior_by_position:
        group_position = (
            df.sort_values(by=["season", "gameweek"])
            .groupby(group_col)["position"]
            .last()
        )
        prior_sum = df.groupby("position")[mean_col].mean() * n_prior
        return (group_sums + prior_sum.loc[group_position].values) / (
            group_counts + n_prior
        )

    overall_prior = n_prior * float(df[mean_col].mean())
    return (group_sums + overall_prior) / (group_counts + n_prior)


def points_by_appearance_length(
    fitted: tuple[pd.Series, pd.Series], player_id: int, minutes: float
) -> float:
    """
    A player's fitted points for this many minutes.

    `fitted` is the per-player means for a full appearance and for a short one;
    a player who did not appear, or who is not in them, gets zero.
    """
    if minutes >= MIN_MINUTES_FULL:
        return float(fitted[0].get(player_id, 0.0))
    if minutes >= MIN_MINUTES_SHORT:
        return float(fitted[1].get(player_id, 0.0))
    return 0.0
