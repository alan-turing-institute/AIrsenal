"""Shrinking a player's own history towards the average for their position."""

import pandas as pd


def mean_group_prior(
    df: pd.DataFrame,
    group_col: str,
    mean_col: str,
    n_prior: int = 10,
    prior_by_position: bool = False,
) -> pd.Series:
    """Compute empirical Bayes group means with a prior weight."""
    group_counts = df.groupby(group_col)[mean_col].count()
    group_sums = df.groupby(group_col)[mean_col].sum()
    group_position = (
        df.sort_values(by=["season", "gameweek"]).groupby(group_col)["position"].last()
    )

    if prior_by_position:
        prior_sum = df.groupby("position")[mean_col].mean() * n_prior
        return (group_sums + prior_sum.loc[group_position].values) / (
            group_counts + n_prior
        )

    overall_prior = n_prior * float(df[mean_col].mean())
    return (group_sums + overall_prior) / (group_counts + n_prior)
