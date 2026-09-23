"""Points for defensive contributions, awarded from the 25/26 season."""

import pandas as pd
from sqlalchemy.orm import Session

from airsenal.db.queries.scores import get_player_scores_df
from airsenal.game.enums import Position
from airsenal.game.scoring import (
    MAX_MINUTES_MATCH,
    MIN_MINUTES_FULL,
    MIN_MINUTES_SHORT,
    def_cons_required,
    points_for_def_cons,
)
from airsenal.prediction.point_components.empirical_bayes import (
    mean_group_prior,
    points_by_appearance_length,
)
from airsenal.prediction.protocols import ComponentRequest


def fit_def_con(
    gameweek: int,
    season: str,
    n_prior: int = 10,
    dbsession: Session | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Fit defensive contribution points model across positions."""

    def get_def_con_df(min_minutes: int, max_minutes: int) -> pd.Series:
        dfs = []
        for position in (Position.DEF, Position.MID, Position.FWD):
            df = get_player_scores_df(
                min_minutes=min_minutes,
                max_minutes=max_minutes,
                position=position,
                gameweek=gameweek,
                season=season,
                dbsession=dbsession,
            ).dropna(subset="defensive_contribution")
            df["def_con_pts"] = (
                df["defensive_contribution"] >= def_cons_required[position]
            ).astype(int) * points_for_def_cons
            dfs.append(df)

        return mean_group_prior(
            pd.concat(dfs),
            "player_id",
            "def_con_pts",
            n_prior=n_prior,
            prior_by_position=True,
        )

    df_90 = get_def_con_df(MIN_MINUTES_FULL, MAX_MINUTES_MATCH)
    df_60 = get_def_con_df(MIN_MINUTES_SHORT, MIN_MINUTES_FULL - 1)

    return (df_90, df_60)


class DefConComponent:
    """
    How often this player clears the defensive-contribution threshold.

    Goalkeepers cannot: `def_cons_required` puts their threshold out of reach,
    because FPL does not award them these points.
    """

    name = "def_con"

    def __init__(self, n_prior: int = 10):
        self.n_prior = n_prior
        self.fitted: tuple[pd.Series, pd.Series] | None = None

    def fit(self, gameweek: int, season: str, dbsession: Session) -> "DefConComponent":
        self.fitted = fit_def_con(
            gameweek, season, n_prior=self.n_prior, dbsession=dbsession
        )
        return self

    def expected_points(self, request: ComponentRequest) -> float:
        if self.fitted is None:
            msg = "The defensive contribution component has not been fitted yet."
            raise RuntimeError(msg)
        return points_by_appearance_length(
            self.fitted, request.player_id, request.minutes
        )
