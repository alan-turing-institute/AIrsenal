"""
Model for predicting a player's expected minutes in an upcoming fixture.

Replaces the previous approach of averaging points across a player's own last few
match minutes (see git history of prediction_utils.calc_predicted_points_for_player).
That approach has no way to notice that a player's own recent minutes were low only
because a same-team, same-position competitor was fit and picked ahead of them - if
that competitor is now injured or suspended, the player is likely to inherit
significant minutes despite a zero-heavy recent history, and the old approach would
still predict zero.

This model adds that missing signal: for each player/fixture, in addition to their own
recent minutes, it uses the summed "typical minutes when playing" of same-team, same-
(FPL-)position teammates who are currently unavailable - so a nailed-on starter being
absent counts for much more than a fringe player being absent. Trained on past-season
PlayerScore/Absence data via build_minutes_training_data(). Note FPL positions are
coarse (e.g. left-back and centre-back aren't distinguished), so "same position" is an
approximation of "direct competitor for the same starting slot", not a precise depth
chart - the best available given the data, not a precise one.

Predicting a single number that's bimodal in reality (most matches a player plays
either ~0 or ~90 minutes) with one regressor tends to regress every prediction toward
the middle, since squared-error loss is minimised by the conditional mean, not by
being confidently right about the common cases (confirmed via backtest - see
notebooks/minutes_model_2526_backtest.ipynb). MinutesModel is instead a two-stage
model: a classifier over minutes buckets (0, 1-59, 60-89, 90) - matching the actual
kinks in the points-scoring rules, see FPL_scoring_rules.get_appearance_points - plus a
regressor within each of the two partial buckets, combined into one continuous number
as a probability-weighted mixture (see MinutesModel.predict). This keeps the single
"drop-in" output prediction_utils.py relies on, while letting the model stay confident
near 0/90 for the (large) majority of "obvious" rows.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sqlalchemy import select
from sqlalchemy.orm.session import Session

from airsenal.framework.schema import Player, PlayerAttributes, session
from airsenal.framework.utils import (
    CURRENT_SEASON,
    NEXT_GAMEWEEK,
    get_recent_minutes_for_player,
    get_recent_playerscore_rows,
    list_players,
    was_historic_absence,
)

FEATURE_COLUMNS = ["own_recent_minutes", "absent_teammates_typical_minutes", "position"]
OWN_RECENT_MINUTES_WINDOW = 3
TEAMMATE_TYPICAL_MINUTES_WINDOW = 8

MINUTES_BUCKETS = ["0", "1-59", "60-89", "90"]
_MINUTES_BUCKET_BINS = [-1, 0, 59, 89, 200]


def _minutes_bucket(minutes: pd.Series) -> pd.Series:
    return pd.cut(minutes, bins=_MINUTES_BUCKET_BINS, labels=MINUTES_BUCKETS)


class MinutesModel:
    """Classifier over minutes buckets (0, 1-59, 60-89, 90) plus a regressor within
    each of the two partial buckets, combined into a single expected-minutes number
    (see module docstring for why). Feature-building logic is kept alongside the
    models themselves so train and predict time can't drift apart.
    """

    def __init__(
        self,
        classifier: HistGradientBoostingClassifier | None = None,
        regressor_low: HistGradientBoostingRegressor | None = None,
        regressor_high: HistGradientBoostingRegressor | None = None,
    ) -> None:
        self.classifier = classifier or HistGradientBoostingClassifier(
            categorical_features=["position"], random_state=42
        )
        self.regressor_low = regressor_low or HistGradientBoostingRegressor(
            categorical_features=["position"], random_state=42
        )
        self.regressor_high = regressor_high or HistGradientBoostingRegressor(
            categorical_features=["position"], random_state=42
        )

    def fit(self, df: pd.DataFrame) -> "MinutesModel":
        """df must have the FEATURE_COLUMNS plus a 'minutes' target column - see
        build_minutes_training_data().
        """
        x = _prepare_feature_frame(df[FEATURE_COLUMNS])
        bucket = _minutes_bucket(df["minutes"])
        self.classifier.fit(x, bucket)

        low_mask = (bucket == "1-59").to_numpy()
        high_mask = (bucket == "60-89").to_numpy()
        self.regressor_low.fit(x[low_mask], df["minutes"][low_mask])
        self.regressor_high.fit(x[high_mask], df["minutes"][high_mask])
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Vectorised prediction for many rows at once - df must have the
        FEATURE_COLUMNS (see predict_one for a single-row convenience wrapper).

        Returns a probability-weighted mixture of each bucket's point estimate
        (0.0 / regressor_low / regressor_high / 90.0), not a hard classification -
        this is what lets the output stay a single continuous number.
        """
        x = _prepare_feature_frame(df[FEATURE_COLUMNS])
        proba = self.classifier.predict_proba(x)
        classes = self.classifier.classes_

        low_pred = self.regressor_low.predict(x)
        high_pred = self.regressor_high.predict(x)
        point_estimates = {
            "0": np.zeros(len(x)),
            "1-59": low_pred,
            "60-89": high_pred,
            "90": np.full(len(x), 90.0),
        }
        estimates = np.column_stack([point_estimates[c] for c in classes])
        expected = (proba * estimates).sum(axis=1)
        return np.clip(expected, 0.0, 90.0)

    def predict_one(
        self,
        own_recent_minutes: float,
        absent_teammates_typical_minutes: float,
        position: str | None,
    ) -> float:
        df = pd.DataFrame(
            {
                "own_recent_minutes": [own_recent_minutes],
                "absent_teammates_typical_minutes": [absent_teammates_typical_minutes],
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


def get_teammate_typical_minutes(
    teammate: Player,
    season: str,
    current_gw: int,
    dbsession: Session = session,
) -> float:
    """Mean minutes played across `teammate`'s last TEAMMATE_TYPICAL_MINUTES_WINDOW
    appearances in which they actually featured (minutes > 0). Zero-minute rows are
    dropped before averaging (rather than just taking their literal last N matches),
    so a teammate who's been out for a few weeks still reports their pre-absence
    playing time instead of decaying toward 0 - the whole point of this feature is to
    weight an absence by how much game time it's likely to free up.
    """
    recent = get_recent_playerscore_rows(
        teammate,
        num_match_to_use=TEAMMATE_TYPICAL_MINUTES_WINDOW,
        season=season,
        last_gw=current_gw - 1,
        dbsession=dbsession,
    )
    minutes_played = [float(r.minutes) for r in recent if r.minutes > 0]
    return float(np.mean(minutes_played)) if minutes_played else 0.0


def sum_absent_teammates_typical_minutes(
    teammates: list[Player],
    season: str,
    current_gw: int,
    fixture_gw: int,
    dbsession: Session = session,
) -> float:
    """Sum of get_teammate_typical_minutes() over the subset of `teammates` who are
    unavailable for the fixture at `fixture_gw`, as known as of `current_gw`. Mirrors
    the same current-vs-historic dispatch already used for the predicted player
    themselves in prediction_utils.calc_predicted_points_for_player
    (is_injured_or_suspended for the live current season, was_historic_absence for
    completed seasons) - applied to teammates here instead of the player being
    predicted.
    """
    total = 0.0
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
            total += get_teammate_typical_minutes(
                teammate, season, current_gw, dbsession=dbsession
            )
    return total


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
    absent_teammates_typical_minutes = sum_absent_teammates_typical_minutes(
        teammates, season, current_gw, fixture_gw, dbsession=dbsession
    )
    return model.predict_one(
        own_recent_minutes, absent_teammates_typical_minutes, position
    )


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
    """Build a (player_id, season, gameweek, own_recent_minutes,
    absent_teammates_typical_minutes, position, minutes) frame from PlayerScore/
    Absence data up to (season, gameweek). Uses bulk queries plus vectorised pandas
    operations throughout, rather than a query per player-match row, since this gets
    rerun on every prediction run (matching how the existing player/team models are
    refit each time rather than persisted - see player_model.py/
    prediction_utils.fit_player_data).

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

    def _rolling_typical_minutes_when_playing(s: pd.Series) -> pd.Series:
        # mask zero-minute rows to NaN first - rolling().mean() ignores NaN entries
        # within the window, so this naturally skips rest/injury weeks rather than
        # decaying toward 0 the way own_recent_minutes above would.
        masked = s.where(s > 0)
        return (
            masked.shift(1)
            .rolling(window=TEAMMATE_TYPICAL_MINUTES_WINDOW, min_periods=1)
            .mean()
        )

    history["typical_minutes_when_playing"] = (
        history.groupby("player_id")["minutes"]
        .transform(_rolling_typical_minutes_when_playing)
        .fillna(0.0)
    )

    # for each (team, position, season, gameweek), the typical-minutes-when-playing
    # of each absent player - used below to weight *other* players' absences for
    # each row by how significant they are, not just count them.
    absentees = (
        history[history["absence_reason"].notna()]
        .groupby(["team", "position", "season", "gameweek"])
        .apply(
            lambda g: dict(
                zip(g["player_id"], g["typical_minutes_when_playing"], strict=True)
            ),
            include_groups=False,
        )
    )

    def _absent_teammates_typical_minutes(row: pd.Series) -> float:
        absent_here = absentees.get(
            (row["team"], row["position"], row["season"], row["gameweek"]), {}
        )
        return sum(v for pid, v in absent_here.items() if pid != row["player_id"])

    # training examples: only rows where the player themselves was available - the
    # only scenario the model is ever actually consulted for (see
    # calc_predicted_points_for_player's is_injured_or_suspended/was_historic_absence
    # short-circuits, which handle the player's own absence deterministically
    # upstream and never call into this model in that case).
    train_df = history[history["absence_reason"].isna()].copy()
    train_df["absent_teammates_typical_minutes"] = train_df.apply(
        _absent_teammates_typical_minutes, axis=1
    )

    return train_df[
        ["player_id", "season", "gameweek", *FEATURE_COLUMNS, "minutes"]
    ].reset_index(drop=True)


def build_minutes_training_data(
    season: str = CURRENT_SEASON,
    gameweek: int = NEXT_GAMEWEEK,
    dbsession: Session = session,
) -> pd.DataFrame:
    """Build a (own_recent_minutes, absent_teammates_typical_minutes, position,
    minutes) training frame from completed-season PlayerScore/Absence data up to
    (season, gameweek). See build_minutes_feature_frame() for the underlying feature
    engineering - this just excludes the current (in-progress) season, which has no
    reliable retrospective Absence record (matches was_historic_absence's own
    restriction to non-current seasons), and drops the player_id/season/gameweek
    columns that a training set doesn't need.
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
