"""
Turning raw goal involvements into the counts a player model is fitted to.

Shared by more than one model, and by `features.py` when it assembles the
training data, so it lives beside the models rather than inside one of them.
"""

import numpy as np
import numpy.typing as npt
import pandas as pd

type FloatArray = npt.NDArray[np.float64]


def get_empirical_bayes_estimates(
    df_emp: pd.DataFrame, prior_goals: float | None = None
) -> FloatArray:
    """
    Get values to use either for Dirichlet prior alphas in the original Stan and numpyro
    player models. Returns number of goals, assists and neither scaled by the
    proportion of minutes & no. matches a player is involved in. If df_emp contains more
    than one player, result is average across all players.

    If prior_goals is not None, normalise the returned alpha values to sum to
    prior_goals.
    """
    # for compatibility with models we zero pad data so all players have
    # the same number of rows (matches). Remove the dummy matches:
    df = df_emp.copy()
    df = df[df["match_id"] != 0]

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
    alpha = np.array([a0, a1, a2])
    if prior_goals is not None:
        alpha = prior_goals * (alpha / alpha.sum())
    return alpha


def scale_goals_by_minutes(
    goals: np.ndarray,
    minutes: np.ndarray,
    time_diff: np.ndarray | None = None,
    epsilon: float | None = None,
    rescale_weights: bool = True,
) -> FloatArray:
    """
    Scale player goal involvements by the proportion of minutes they played
    (specifically: reduce the number of "neither" goals where the player is said
    to have had no involvement.
    goals: np.array with shape (n_players, n_matches, 3) where last axis is no. goals,
    no. assists, and no. goals not involved in
    minutes: np.array with shape (n_players, m_matches)
    time_diff: np.array with shape (n_players, m_matches)
    epsilon: float for weight decay rate with time
    rescale_weights: bool indicating whether to rescale weights to sum to n_matches for
    each player (n_matches the player appeared in where a goal was scored)
    """
    if epsilon is not None and time_diff is None:
        msg = "time_diff must be provided if using time weighting."
        raise ValueError(msg)
    if time_diff is not None and epsilon is not None:
        weights = np.exp(-epsilon * time_diff)
    else:
        weights = np.ones_like(minutes)
    select_matches = (goals.sum(axis=2) > 0) & (minutes > 0)
    n_players, _, _ = goals.shape
    scaled_goals = np.zeros((n_players, 3))
    for p in range(n_players):
        if select_matches[p, :].sum() == 0:
            # player not involved in any matches with goals
            scaled_goals[p, :] = [0, 0, 0]
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

    # players with high goal involvements in few matches may end up with a scaled
    # neither count less than 0 - set these to zero
    scaled_goals[scaled_goals < 0] = 0

    return scaled_goals
