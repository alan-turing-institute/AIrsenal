"""The points a player loses to yellow and red cards."""

import pandas as pd
from sqlalchemy.orm import Session

from airsenal.db.queries.scores import get_player_scores_df
from airsenal.game.scoring import (
    MIN_MINUTES_SHORT,
    points_for_red_card,
    points_for_yellow_card,
)
from airsenal.prediction.point_components.empirical_bayes import mean_group_prior
from airsenal.prediction.protocols import ComponentRequest


def fit_card_points(
    gameweek: int,
    season: str,
    n_prior: int = 10,
    min_minutes: int = 1,
    dbsession: Session | None = None,
) -> pd.Series:
    """Fit card penalty points model using historical player scores."""
    df = get_player_scores_df(
        min_minutes=min_minutes, gameweek=gameweek, season=season, dbsession=dbsession
    )

    df["card_pts"] = (
        points_for_yellow_card * df["yellow_cards"]
        + points_for_red_card * df["red_cards"]
    )

    return mean_group_prior(
        df, "player_id", "card_pts", n_prior=n_prior, prior_by_position=False
    )


class CardComponent:
    """How often this player is booked, shrunk towards the league average."""

    name = "cards"

    def __init__(self, n_prior: int = 10):
        self.n_prior = n_prior
        self.fitted: pd.Series | None = None

    def fit(self, gameweek: int, season: str, dbsession: Session) -> "CardComponent":
        self.fitted = fit_card_points(
            gameweek, season, n_prior=self.n_prior, dbsession=dbsession
        )
        return self

    def expected_points(self, request: ComponentRequest) -> float:
        if self.fitted is None:
            msg = "The card component has not been fitted yet."
            raise RuntimeError(msg)
        if request.minutes >= MIN_MINUTES_SHORT:
            return float(self.fitted.get(request.player_id, 0.0))
        return 0.0
