"""A goalkeeper's points for saves."""

import pandas as pd
from sqlalchemy.orm import Session

from airsenal.db.queries.scores import get_player_scores_df
from airsenal.game.enums import Position
from airsenal.game.scoring import (
    MAX_MINUTES_MATCH,
    MIN_MINUTES_FULL,
    saves_for_point,
)
from airsenal.prediction.point_components.empirical_bayes import mean_group_prior
from airsenal.prediction.protocols import ComponentRequest


def fit_save_points(
    gameweek: int,
    season: str,
    n_prior: int = 10,
    min_minutes: int = MAX_MINUTES_MATCH,
    dbsession: Session | None = None,
) -> pd.Series:
    """Fit goalkeeper save points model using historical player scores."""
    df = get_player_scores_df(
        min_minutes=min_minutes,
        position=Position.GK,
        gameweek=gameweek,
        season=season,
        dbsession=dbsession,
    )

    df["save_pts"] = (df["saves"] / saves_for_point).astype(int)

    return mean_group_prior(df, "player_id", "save_pts", n_prior=n_prior)


class SaveComponent:
    """
    A point per three saves, from what this keeper has averaged.

    Fitted on full matches only, and awarded only for one: a keeper who came on
    at half time is not expected to make a full match's saves.
    """

    name = "saves"

    def __init__(self, n_prior: int = 10):
        self.n_prior = n_prior
        self.fitted: pd.Series | None = None

    def fit(self, gameweek: int, season: str, dbsession: Session) -> "SaveComponent":
        self.fitted = fit_save_points(
            gameweek, season, n_prior=self.n_prior, dbsession=dbsession
        )
        return self

    def expected_points(self, request: ComponentRequest) -> float:
        if self.fitted is None:
            msg = "The save component has not been fitted yet."
            raise RuntimeError(msg)
        if request.position != Position.GK:
            return 0.0
        if request.minutes >= MIN_MINUTES_FULL:
            return float(self.fitted.get(request.player_id, 0.0))
        return 0.0
