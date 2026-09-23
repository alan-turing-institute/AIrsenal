"""Turning raw goal involvements into the counts a player model is fitted to."""

import numpy as np
import numpy.typing as npt
import pandas as pd

type FloatArray = npt.NDArray[np.float64]


def get_empirical_bayes_estimates(df_emp: pd.DataFrame) -> FloatArray:
    """
    Dirichlet prior alphas: goals, assists, and neither.

    Each is scaled by the proportion of minutes and matches the player was
    involved in.

    Args:
        df_emp: One player, or several - several are averaged into a single set
            of alphas.
    """
    # drop the padding rows that give every player the same number of matches
    df = df_emp[df_emp["match_id"] != 0]

    player_goals = df["goals"].sum()
    player_assists = df["assists"].sum()
    player_neither = df["neither"].sum()
    player_minutes = df["minutes"].sum()
    team_goals = df["team_goals"].sum()
    total_minutes = 90 * len(df)
    n_matches = df.groupby("player_name").count()["goals"].mean()

    # Total no. of player goals, assists, neither:
    # no. matches played * fraction goals scored * (1 / fraction mins played)
    a0 = n_matches * (player_goals / team_goals) * (total_minutes / player_minutes)
    a1 = n_matches * (player_assists / team_goals) * (total_minutes / player_minutes)
    a2 = (
        n_matches
        * (
            (player_neither / team_goals)
            - (total_minutes - player_minutes) / total_minutes
        )
        * (total_minutes / player_minutes)
    )
    return np.array([a0, a1, a2])


def scale_goals_by_minutes(
    goals: np.ndarray,
    minutes: np.ndarray,
    time_diff: np.ndarray | None = None,
    epsilon: float | None = None,
    rescale_weights: bool = True,
) -> FloatArray:
    """
    Scale goal involvements by the proportion of minutes a player played.

    What shrinks is the "neither" count - goals the player is said to have had no
    involvement in. A negative scaled count is clipped to zero.

    Args:
        goals: Shape (n_players, n_matches, 3), the last axis being goals,
            assists, and goals the player was not involved in.
        minutes: Shape (n_players, n_matches).
        time_diff: Shape (n_players, n_matches). Required if `epsilon` is given.
        epsilon: Weight decay rate with time. None means no time weighting.
        rescale_weights: If True, rescale each player's weights to sum to the
            number of matches they appeared in where a goal was scored.
    """
    if epsilon is None:
        weights = np.ones_like(minutes)
    elif time_diff is None:
        msg = "time_diff must be provided if using time weighting."
        raise ValueError(msg)
    else:
        weights = np.exp(-epsilon * time_diff)
    select_matches = (goals.sum(axis=2) > 0) & (minutes > 0)
    n_players, _, _ = goals.shape
    scaled_goals = np.zeros((n_players, 3))
    for p in range(n_players):
        if select_matches[p, :].sum() == 0:
            # player not involved in any matches with goals, so stays at zero
            continue

        match_weights = weights[p, select_matches[p, :]]
        if rescale_weights:
            match_weights = (
                select_matches[p, :].sum() * match_weights / match_weights.sum()
            )
        team_goals = (
            goals[p, select_matches[p, :], :].sum(axis=1) * match_weights
        ).sum()
        team_mins = (90 * match_weights).sum()
        player_mins = (minutes[p, select_matches[p, :]] * match_weights).sum()
        player_goals = (goals[p, select_matches[p, :], 0] * match_weights).sum()
        player_assists = (goals[p, select_matches[p, :], 1] * match_weights).sum()
        player_neither = (
            team_goals * (player_mins / team_mins) - player_goals - player_assists
        )
        scaled_goals[p, :] = [player_goals, player_assists, player_neither]

    # substituted players with high goal involvements in few matches may end up with a
    # scaled neither count less than 0 - set these to zero
    scaled_goals[scaled_goals < 0] = 0

    return scaled_goals
