"""
Model for predicting a player's expected minutes in an upcoming fixture.

Replaces the previous approach of averaging points across a player's own last few
match minutes (see git history of prediction_utils.calc_predicted_points_for_player).
That approach has no way to notice that a player's own recent minutes were low only
because a same-team, same-position competitor was fit and picked ahead of them - if
that competitor is now injured or suspended, the player is likely to inherit
significant minutes despite a zero-heavy recent history, and the old approach would
still predict zero.

This model adds that missing signal: for each player/fixture, in addition to their
own recent minutes, it uses a count of same-team, same-(FPL-)position teammates who
are currently unavailable. Trained on past-season PlayerScore/Absence data via
build_minutes_training_data(). Note FPL positions are coarse (e.g. left-back and
centre-back aren't distinguished), so "same position" is an approximation of "direct
competitor for the same starting slot", not a precise depth chart - the best
available given the data, not a precise one.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sqlalchemy import select
from sqlalchemy.orm.session import Session

from airsenal.framework.schema import Player, PlayerAttributes, session
from airsenal.framework.utils import (
    CURRENT_SEASON,
    NEXT_GAMEWEEK,
    get_recent_minutes_for_player,
    list_players,
    was_historic_absence,
)

FEATURE_COLUMNS = ["own_recent_minutes", "n_teammates_absent", "position"]
OWN_RECENT_MINUTES_WINDOW = 3


class MinutesModel:
    """Thin wrapper around a fitted regressor plus the feature-building logic needed
    at both train and predict time (kept together so the two can't drift apart).
    """

    def __init__(self, estimator: HistGradientBoostingRegressor | None = None) -> None:
        self.estimator = estimator or HistGradientBoostingRegressor(
            categorical_features=["position"], random_state=42
        )

    def fit(self, df: pd.DataFrame) -> "MinutesModel":
        """df must have the FEATURE_COLUMNS plus a 'minutes' target column - see
        build_minutes_training_data().
        """
        x = _prepare_feature_frame(df[FEATURE_COLUMNS])
        self.estimator.fit(x, df["minutes"])
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Vectorised prediction for many rows at once - df must have the
        FEATURE_COLUMNS (see predict_one for a single-row convenience wrapper).
        """
        x = _prepare_feature_frame(df[FEATURE_COLUMNS])
        return np.clip(self.estimator.predict(x), 0.0, 90.0)

    def predict_one(
        self, own_recent_minutes: float, n_teammates_absent: int, position: str | None
    ) -> float:
        df = pd.DataFrame(
            {
                "own_recent_minutes": [own_recent_minutes],
                "n_teammates_absent": [n_teammates_absent],
                "position": [position],
            }
        )
        return float(self.predict(df)[0])


def _prepare_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["position"] = df["position"].astype("category")
    return df


def get_position_teammates(
    player: Player, season: str, gameweek: int, dbsession: Session = session
) -> list[Player]:
    """Other players in the same team and (FPL) position as `player` - the closest
    approximation of "direct competitors for the same starting slot" available.
    """
    team = player.team(season, gameweek)
    position = player.position(season)
    if team is None or position is None:
        return []
    teammates = list_players(
        position=position,
        team=team,
        season=season,
        gameweek=gameweek,
        dbsession=dbsession,
    )
    return [p for p in teammates if p.player_id != player.player_id]


def count_absent_teammates(
    teammates: list[Player],
    season: str,
    current_gw: int,
    fixture_gw: int,
    dbsession: Session = session,
) -> int:
    """How many of `teammates` are unavailable for the fixture at `fixture_gw`, as
    known as of `current_gw`. Mirrors the same current-vs-historic dispatch already
    used for the predicted player themselves in
    prediction_utils.calc_predicted_points_for_player (is_injured_or_suspended for
    the live current season, was_historic_absence for completed seasons) - applied to
    teammates here instead of the player being predicted.
    """
    count = 0
    for teammate in teammates:
        if season == CURRENT_SEASON:
            unavailable = teammate.is_injured_or_suspended(
                season, current_gw, fixture_gw
            )
        else:
            unavailable = was_historic_absence(
                teammate, gameweek=fixture_gw, season=season, dbsession=dbsession
            )
        if unavailable:
            count += 1
    return count


def predict_expected_minutes(
    model: MinutesModel,
    player: Player,
    season: str,
    current_gw: int,
    fixture_gw: int,
    own_recent_minutes: float | None = None,
    dbsession: Session = session,
) -> float:
    """Predict expected minutes for `player` at `fixture_gw`. `own_recent_minutes`
    can be precomputed once and passed in when predicting several fixtures for the
    same player/current_gw in one call (it doesn't vary match to match, unlike
    teammate availability) to avoid redundant querying - computed here if not given.
    """
    if own_recent_minutes is None:
        recent = get_recent_minutes_for_player(
            player,
            num_match_to_use=OWN_RECENT_MINUTES_WINDOW,
            season=season,
            last_gw=current_gw - 1,
            dbsession=dbsession,
        )
        own_recent_minutes = float(np.mean(recent)) if recent else 0.0

    position = player.position(season)
    teammates = get_position_teammates(player, season, current_gw, dbsession=dbsession)
    n_absent = count_absent_teammates(
        teammates, season, current_gw, fixture_gw, dbsession=dbsession
    )
    return model.predict_one(own_recent_minutes, n_absent, position)


def _fetch_team_position_lookup(seasons: list[str], dbsession: Session) -> pd.DataFrame:
    """Bulk (player_id, season, gameweek) -> (team, position) lookup, to merge into
    a history dataframe - avoids the per-row DB queries that Player.team()/
    Player.position() would do if called once per training row.
    """
    rows = dbsession.execute(
        select(
            PlayerAttributes.player_id,
            PlayerAttributes.season,
            PlayerAttributes.gameweek,
            PlayerAttributes.team,
            PlayerAttributes.position,
        ).where(PlayerAttributes.season.in_(seasons))
    ).all()
    return pd.DataFrame(
        rows, columns=["player_id", "season", "gameweek", "team", "position"]
    )


def build_minutes_feature_frame(
    season: str = CURRENT_SEASON,
    gameweek: int = NEXT_GAMEWEEK,
    dbsession: Session = session,
) -> pd.DataFrame:
    """Build a (player_id, season, gameweek, own_recent_minutes, n_teammates_absent,
    position, minutes) frame from PlayerScore/Absence data up to (season, gameweek).
    Uses bulk queries plus vectorised pandas operations throughout, rather than a
    query per player-match row, since this gets rerun on every prediction run
    (matching how the existing player/team models are refit each time rather than
    persisted - see player_model.py/prediction_utils.fit_player_data).

    Keeps player_id/season/gameweek (unlike build_minutes_training_data's
    FEATURE_COLUMNS-only output) so callers can split rows by season themselves -
    e.g. to hold out a specific season for backtesting.
    """
    # imported here, not at module level, to avoid a circular import - prediction_utils
    # imports MinutesModel/predict_expected_minutes from this module.
    from airsenal.framework.prediction_utils import (  # noqa: PLC0415
        get_player_history_df,
    )

    history = get_player_history_df(
        all_players=True,
        fill_blank=False,
        season=season,
        gameweek=gameweek,
        dbsession=dbsession,
    )
    if history.empty:
        return pd.DataFrame(
            columns=["player_id", "season", "gameweek", *FEATURE_COLUMNS, "minutes"]
        )

    team_position = _fetch_team_position_lookup(
        history["season"].unique().tolist(), dbsession
    )
    history = history.merge(
        team_position, on=["player_id", "season", "gameweek"], how="inner"
    )

    history = history.sort_values(["player_id", "season", "gameweek"])

    def _rolling_own_minutes(s: pd.Series) -> pd.Series:
        return (
            s.shift(1).rolling(window=OWN_RECENT_MINUTES_WINDOW, min_periods=1).mean()
        )

    history["own_recent_minutes"] = history.groupby("player_id")["minutes"].transform(
        _rolling_own_minutes
    )

    # for each (team, position, season, gameweek), which players were absent - used
    # below to count *other* players' absences for each row, i.e. competitors for
    # the same slot, not the row's own player.
    absentees = (
        history[history["absence_reason"].notna()]
        .groupby(["team", "position", "season", "gameweek"])["player_id"]
        .apply(set)
    )

    def _count_absent_teammates(row: Any) -> int:
        absent_here = absentees.get(
            (row["team"], row["position"], row["season"], row["gameweek"]), set()
        )
        return len(absent_here - {row["player_id"]})

    # training examples: only rows where the player themselves was available - the
    # only scenario the model is ever actually consulted for (see
    # calc_predicted_points_for_player's is_injured_or_suspended/was_historic_absence
    # short-circuits, which handle the player's own absence deterministically
    # upstream and never call into this model in that case).
    train_df = history[history["absence_reason"].isna()].copy()
    train_df["n_teammates_absent"] = train_df.apply(_count_absent_teammates, axis=1)

    return train_df[
        ["player_id", "season", "gameweek", *FEATURE_COLUMNS, "minutes"]
    ].reset_index(drop=True)


def build_minutes_training_data(
    season: str = CURRENT_SEASON,
    gameweek: int = NEXT_GAMEWEEK,
    dbsession: Session = session,
) -> pd.DataFrame:
    """Build a (own_recent_minutes, n_teammates_absent, position, minutes) training
    frame from completed-season PlayerScore/Absence data up to (season, gameweek).
    See build_minutes_feature_frame() for the underlying feature engineering - this
    just excludes the current (in-progress) season, which has no reliable
    retrospective Absence record (matches was_historic_absence's own restriction to
    non-current seasons), and drops the player_id/season/gameweek columns that a
    training set doesn't need.
    """
    df = build_minutes_feature_frame(
        season=season, gameweek=gameweek, dbsession=dbsession
    )
    df = df[df["season"] != CURRENT_SEASON]
    return df[[*FEATURE_COLUMNS, "minutes"]].reset_index(drop=True)


def fit_minutes_model(
    season: str = CURRENT_SEASON,
    gameweek: int = NEXT_GAMEWEEK,
    dbsession: Session = session,
) -> MinutesModel:
    """Build training data up to (season, gameweek) and fit a fresh MinutesModel."""
    df = build_minutes_training_data(
        season=season, gameweek=gameweek, dbsession=dbsession
    )
    return MinutesModel().fit(df)
