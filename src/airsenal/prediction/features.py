"""Assembling the historical data the models are fitted to."""

from collections import defaultdict

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from airsenal.core.console import track
from airsenal.core.logging import get_logger
from airsenal.db.models import Absence, PlayerAttributes, PlayerScore
from airsenal.db.queries.fixtures import get_fixtures_for_gameweeks
from airsenal.db.queries.gameweeks import is_future_gameweek, next_gameweek
from airsenal.db.queries.players import get_max_matches_per_player, list_players
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
from airsenal.prediction.player_models.scaling import get_empirical_bayes_estimates
from airsenal.prediction.protocols import PlayerFitData

logger = get_logger(__name__)


def get_player_history_df(
    position: str = "all",
    all_players: bool = False,
    fill_blank: bool = True,
    season: str = CURRENT_SEASON,
    gameweek: int | None = None,
    dbsession: Session | None = None,
) -> pd.DataFrame:
    """Fetch historical player performance data and build a structured DataFrame."""
    gameweek = next_gameweek() if gameweek is None else gameweek
    dbsession = dbsession if dbsession is not None else get_session()
    col_names = [
        "player_id",
        "player_name",
        "match_id",
        "date",
        "season",
        "gameweek",
        "goals",
        "assists",
        "minutes",
        "team_goals",
        "expected_goals",
        "expected_assists",
        "absence_reason",
        "absence_detail",
    ]
    player_data = []

    if all_players:
        q = dbsession.scalars(
            select(PlayerAttributes).options(selectinload(PlayerAttributes.player))
        )
        players = []
        seen_player_ids = set()
        for p in q:
            if p.player_id in seen_player_ids:
                continue
            seen_player_ids.add(p.player_id)
            players.append(p.player)
    else:
        players = list_players(
            position=position, season=season, gameweek=gameweek, dbsession=dbsession
        )

    player_ids = [p.player_id for p in players]
    scores_by_player = defaultdict(list)
    absences_by_player_season = defaultdict(list)

    if player_ids:
        all_scores = dbsession.scalars(
            select(PlayerScore)
            .options(
                selectinload(PlayerScore.fixture),
                selectinload(PlayerScore.result),
            )
            .where(PlayerScore.player_id.in_(player_ids))
        ).all()
        for score in all_scores:
            scores_by_player[score.player_id].append(score)

        score_seasons = {score.fixture.season for score in all_scores}
        if score_seasons:
            absences = dbsession.scalars(
                select(Absence)
                .where(
                    Absence.player_id.in_(player_ids),
                    Absence.season.in_(score_seasons),
                )
                .order_by(Absence.id)
            ).all()
            for absence in absences:
                if absence.player_id is None:
                    continue
                absences_by_player_season[(absence.player_id, absence.season)].append(
                    absence
                )

    max_matches_per_player = get_max_matches_per_player(
        position, season=season, gameweek=gameweek, dbsession=dbsession
    )
    for player in track(
        players, description=f"Filling player history dataframe for {position}:"
    ):
        results = scores_by_player.get(player.player_id, [])
        row_count = 0
        for row in results:
            if is_future_gameweek(
                row.fixture.season,
                row.fixture.gameweek,
                current_season=season,
                next_gameweek=gameweek,
            ):
                continue

            match_id = row.result_id
            if not match_id:
                logger.warning("Couldn't find result for %s", row.fixture)
                continue

            minutes = row.minutes
            goals = row.goals
            assists = row.assists
            match_result = row.result
            match_date = row.fixture.date

            if row.fixture.home_team == row.opponent:
                team_goals = match_result.away_score
            elif row.fixture.away_team == row.opponent:
                team_goals = match_result.home_score
            else:
                logger.warning("Unknown opponent!")
                team_goals = -1

            expected_goals = row.expected_goals
            expected_assists = row.expected_assists
            matching_absences = [
                ab
                for ab in absences_by_player_season.get(
                    (player.player_id, row.fixture.season), []
                )
                if ab.gw_until is not None
                and row.fixture.gameweek is not None
                and ab.gw_from < row.fixture.gameweek
                and ab.gw_until > row.fixture.gameweek
            ]
            # A single absence is recorded as a scalar rather than a 1-element list,
            # so the resulting dataframe column reads naturally.
            absence_reason: str | list[str] | None = None
            absence_detail: str | list[str | None] | None = None
            if matching_absences:
                reasons = [ab.reason for ab in matching_absences]
                details = [ab.details for ab in matching_absences]
                absence_reason = reasons[0] if len(reasons) == 1 else reasons
                absence_detail = details[0] if len(details) == 1 else details

            player_data.append(
                [
                    player.player_id,
                    player.name,
                    match_id,
                    match_date,
                    row.fixture.season,
                    row.fixture.gameweek,
                    goals,
                    assists,
                    minutes,
                    team_goals,
                    expected_goals,
                    expected_assists,
                    absence_reason,
                    absence_detail,
                ]
            )
            row_count += 1

        if fill_blank and row_count < max_matches_per_player:
            blank_row = [
                player.player_id,
                player.name,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                None,
                None,
            ]
            player_data.extend(
                [list(blank_row) for _ in range(max_matches_per_player - row_count)]
            )

    df = pd.DataFrame(player_data, columns=col_names)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.reset_index(drop=True, inplace=True)

    return df


def process_player_data(
    prefix: str,
    season: str = CURRENT_SEASON,
    gameweek: int | None = None,
    dbsession: Session | None = None,
) -> PlayerFitData:
    """Process and structure historical player data for model fitting."""
    gameweek = next_gameweek() if gameweek is None else gameweek
    dbsession = dbsession if dbsession is not None else get_session()
    df = get_player_history_df(
        prefix, season=season, gameweek=gameweek, dbsession=dbsession
    )
    df["neither"] = df["team_goals"] - df["goals"] - df["assists"]
    df.loc[(df["neither"] < 0), ["neither", "team_goals", "goals", "assists"]] = [
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    alpha = get_empirical_bayes_estimates(df)

    y = (
        df.sort_values("player_id")[["goals", "assists", "neither"]]
        .to_numpy()
        .reshape(
            (
                df["player_id"].nunique(),
                df.groupby("player_id").count().iloc[0]["player_name"],
                3,
            )
        )
    )

    minutes = (
        df.sort_values("player_id")[["minutes"]]
        .to_numpy()
        .reshape(
            (
                df["player_id"].nunique(),
                df.groupby("player_id").count().iloc[0]["player_name"],
            )
        )
    )

    nplayer = df["player_id"].nunique()
    nmatch = df.groupby("player_id").count().iloc[0]["player_name"]
    player_ids = np.sort(df["player_id"].unique())

    now_date = np.array(
        [
            pd.Timestamp(f.date).replace(tzinfo=None).date()
            for f in get_fixtures_for_gameweeks([gameweek], season, dbsession)
            if f.date is not None
        ]
    ).min()

    match_date = df["date"].fillna(df["date"].min()).dt.date
    df["time_diff"] = (now_date - match_date) / pd.Timedelta(days=365)
    time_diff = (
        df.sort_values("player_id")[["time_diff"]]
        .to_numpy()
        .reshape(
            (
                df["player_id"].nunique(),
                df.groupby("player_id").count().iloc[0]["player_name"],
            )
        )
    )

    return {
        "player_ids": player_ids,
        "nplayer": nplayer,
        "nmatch": nmatch,
        "minutes": minutes.astype("int64"),
        "y": y.astype("int64"),
        "alpha": alpha,
        "time_diff": time_diff,
    }


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
            season,
            gameweek,
            min_minutes=min_minutes,
            max_minutes=max_minutes,
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
        season,
        gameweek,
        min_minutes=min_minutes,
        position=Position.GK,
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
        season, gameweek, min_minutes=min_minutes, dbsession=dbsession
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
                season,
                gameweek,
                min_minutes=min_minutes,
                max_minutes=max_minutes,
                position=position,
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
