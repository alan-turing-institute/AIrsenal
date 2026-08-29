"""
The fitted models behind the smaller components of an FPL score.

Bonus points, save points, card points and defensive contributions. Each is an
empirical-Bayes group mean over past performances: a player's own history,
shrunk towards the average for their position by `mean_group_prior`.

`PointsConfig` defines whether each is used by `prediction/run.py`.
"""

import pandas as pd
from sqlalchemy.orm import Session

from airsenal.core.logging import get_logger
from airsenal.db.queries.gameweeks import next_gameweek
from airsenal.db.queries.scores import get_player_scores_df
from airsenal.db.session import get_session
from airsenal.game.enums import Position
from airsenal.game.scoring import (
    MAX_MINUTES_MATCH,
    MIN_MINUTES_FULL,
    MIN_MINUTES_SHORT,
    def_cons_required,
    points_for_def_cons,
    points_for_red_card,
    points_for_yellow_card,
    saves_for_point,
)
from airsenal.game.season import CURRENT_SEASON

logger = get_logger(__name__)


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


def fit_bonus_points(
    gameweek: int | None = None,
    season: str = CURRENT_SEASON,
    n_prior: int = 10,
    dbsession: Session | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Fit bonus points model using historical player scores."""
    gameweek = next_gameweek() if gameweek is None else gameweek
    dbsession = dbsession if dbsession is not None else get_session()

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


def fit_save_points(
    gameweek: int | None = None,
    season: str = CURRENT_SEASON,
    n_prior: int = 10,
    min_minutes: int = MAX_MINUTES_MATCH,
    dbsession: Session | None = None,
) -> pd.Series:
    """Fit goalkeeper save points model using historical player scores."""
    gameweek = next_gameweek() if gameweek is None else gameweek
    dbsession = dbsession if dbsession is not None else get_session()
    df = get_player_scores_df(
        min_minutes=min_minutes,
        position=Position.GK,
        gameweek=gameweek,
        season=season,
        dbsession=dbsession,
    )

    df["save_pts"] = (df["saves"] / saves_for_point).astype(int)

    return mean_group_prior(df, "player_id", "save_pts", n_prior=n_prior)


def fit_card_points(
    gameweek: int | None = None,
    season: str = CURRENT_SEASON,
    n_prior: int = 10,
    min_minutes: int = 1,
    dbsession: Session | None = None,
) -> pd.Series:
    """Fit card penalty points model using historical player scores."""
    gameweek = next_gameweek() if gameweek is None else gameweek
    dbsession = dbsession if dbsession is not None else get_session()
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


def fit_def_con(
    gameweek: int | None = None,
    season: str = CURRENT_SEASON,
    n_prior: int = 10,
    dbsession: Session | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Fit defensive contribution points model across positions."""
    gameweek = next_gameweek() if gameweek is None else gameweek
    dbsession = dbsession if dbsession is not None else get_session()

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
