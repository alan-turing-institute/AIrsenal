"""Bonus points, as a shrunk average of what a player has been given before."""

import pandas as pd
from sqlalchemy.orm import Session

from airsenal.db.queries.scores import get_player_scores_df
from airsenal.game.scoring import (
    MAX_MINUTES_MATCH,
    MIN_MINUTES_FULL,
    MIN_MINUTES_SHORT,
)
from airsenal.prediction.point_components.empirical_bayes import mean_group_prior
from airsenal.prediction.protocols import ComponentRequest


def fit_bonus_points(
    gameweek: int,
    season: str,
    n_prior: int = 10,
    dbsession: Session | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Fit bonus points model using historical player scores."""

    def get_bonus_df(min_minutes: int, max_minutes: int) -> pd.Series:
        df = get_player_scores_df(
            min_minutes=min_minutes,
            max_minutes=max_minutes,
            gameweek=gameweek,
            season=season,
            dbsession=dbsession,
        )
        return mean_group_prior(
            df, "player_id", "bonus", n_prior=n_prior, prior_by_position=True
        )

    df_90 = get_bonus_df(MIN_MINUTES_FULL, MAX_MINUTES_MATCH)
    df_60 = get_bonus_df(MIN_MINUTES_SHORT, MIN_MINUTES_FULL - 1)

    return (df_90, df_60)


class BonusComponent:
    """
    What this player has averaged in bonus points, at two lengths of appearance.

    A player who only came on gets the average for a short appearance, which is
    lower - bonus points go to whoever was on the pitch long enough to earn
    them.
    """

    name = "bonus"

    def __init__(self, n_prior: int = 10):
        self.n_prior = n_prior
        self.fitted: tuple[pd.Series, pd.Series] | None = None

    def fit(self, gameweek: int, season: str, dbsession: Session) -> "BonusComponent":
        self.fitted = fit_bonus_points(
            gameweek, season, n_prior=self.n_prior, dbsession=dbsession
        )
        return self

    def expected_points(self, request: ComponentRequest) -> float:
        if self.fitted is None:
            msg = "The bonus component has not been fitted yet."
            raise RuntimeError(msg)
        if request.minutes >= MIN_MINUTES_FULL:
            return float(self.fitted[0].get(request.player_id, 0.0))
        if request.minutes >= MIN_MINUTES_SHORT:
            return float(self.fitted[1].get(request.player_id, 0.0))
        return 0.0
