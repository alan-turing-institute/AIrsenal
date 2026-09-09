"""Assembling the historical data the models are fitted to."""

from collections import defaultdict
from typing import Any

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
from airsenal.db.queries.scores import get_expected_goals_by_fixture
from airsenal.db.session import get_session
from airsenal.game.enums import Position
from airsenal.game.season import CURRENT_SEASON
from airsenal.prediction.player_models.scaling import get_empirical_bayes_estimates
from airsenal.prediction.protocols import PlayerFitData

logger = get_logger(__name__)

# The columns of the player history frame, in order, and the one place they are
# named. A row is built as a dict rather than a positional list so that adding a
# column - the xG work added three - is an edit here and an edit where the value
# is read off the database, rather than three lists that have to stay in step by
# position.
PLAYER_HISTORY_COLUMNS = (
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
    "team_expected_goals",
    "absence_reason",
    "absence_detail",
)


def blank_player_row(player_id: int, player_name: str) -> dict[str, Any]:
    """
    A padding row, so every player has the same number of matches.

    The models are fitted to rectangular arrays - `process_player_data` reshapes
    to `(nplayer, nmatch, ...)` - so a player with fewer matches than the most
    anyone played is padded out to it. Zero in every column but the player's
    identity and their absences, which is what makes a padding row recognisable:
    `get_empirical_bayes_estimates` drops rows by `match_id == 0`, and a fitted
    model excludes them by `minutes` or by `team_expected_goals` of zero, neither
    of which a real performance has. Deriving the zeros from
    `PLAYER_HISTORY_COLUMNS` means a new column is padded correctly without this
    function being touched.
    """
    return {
        **dict.fromkeys(PLAYER_HISTORY_COLUMNS, 0),
        "player_id": player_id,
        "player_name": player_name,
        "absence_reason": None,
        "absence_detail": None,
    }


def get_player_history_df(
    position: str = "all",
    all_players: bool = False,
    fill_blank: bool = True,
    gameweek: int | None = None,
    season: str = CURRENT_SEASON,
    dbsession: Session | None = None,
) -> pd.DataFrame:
    """Fetch historical player performance data and build a structured DataFrame."""
    gameweek = next_gameweek() if gameweek is None else gameweek
    dbsession = dbsession if dbsession is not None else get_session()
    player_data: list[dict[str, Any]] = []

    if all_players:
        # All of them who play: a manager has attributes and performances like
        # anyone else, and nothing here models one - the same rule as
        # `Position.is_modelled`, asked of the whole set at once.
        q = dbsession.scalars(
            select(PlayerAttributes)
            .where(PlayerAttributes.position.in_(Position.modelled()))
            .options(selectinload(PlayerAttributes.player))
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

    # Per (fixture, team), because an xG involvement is a share of what the
    # whole team was expected to score and this frame holds one position of it.
    team_expected_goals = get_expected_goals_by_fixture(dbsession)

    max_matches_per_player = get_max_matches_per_player(
        position, gameweek=gameweek, season=season, dbsession=dbsession
    )
    for player in track(
        players, description=f"Filling player history dataframe for {position}:"
    ):
        results = scores_by_player.get(player.player_id, [])
        row_count = 0
        for row in results:
            if is_future_gameweek(
                row.fixture.gameweek,
                row.fixture.season,
                current_season=season,
                current_gameweek=gameweek,
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
            team_expected = team_expected_goals.get(
                (row.fixture_id, row.player_team), float("nan")
            )
            matching_absences = [
                ab
                for ab in absences_by_player_season.get(
                    (player.player_id, row.fixture.season), []
                )
                if ab.gameweek_until is not None
                and row.fixture.gameweek is not None
                # Inclusive of gameweek_from, which is the first gameweek missed; see
                # `db.queries.absences.absence_gameweeks`
                and ab.gameweek_from <= row.fixture.gameweek
                and ab.gameweek_until > row.fixture.gameweek
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
                {
                    "player_id": player.player_id,
                    "player_name": player.name,
                    "match_id": match_id,
                    "date": match_date,
                    "season": row.fixture.season,
                    "gameweek": row.fixture.gameweek,
                    "goals": goals,
                    "assists": assists,
                    "minutes": minutes,
                    "team_goals": team_goals,
                    "expected_goals": expected_goals,
                    "expected_assists": expected_assists,
                    "team_expected_goals": team_expected,
                    "absence_reason": absence_reason,
                    "absence_detail": absence_detail,
                }
            )
            row_count += 1

        if fill_blank and row_count < max_matches_per_player:
            player_data.extend(
                blank_player_row(player.player_id, player.name)
                for _ in range(max_matches_per_player - row_count)
            )

    df = pd.DataFrame(player_data, columns=list(PLAYER_HISTORY_COLUMNS))
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.reset_index(drop=True, inplace=True)

    return df


def process_player_data(
    prefix: str,
    gameweek: int | None = None,
    season: str = CURRENT_SEASON,
    dbsession: Session | None = None,
) -> PlayerFitData:
    """Process and structure historical player data for model fitting."""
    gameweek = next_gameweek() if gameweek is None else gameweek
    dbsession = dbsession if dbsession is not None else get_session()
    df = get_player_history_df(
        prefix, gameweek=gameweek, season=season, dbsession=dbsession
    )
    df["neither"] = df["team_goals"] - df["goals"] - df["assists"]
    df.loc[(df["neither"] < 0), ["neither", "team_goals", "goals", "assists"]] = [
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    alpha = get_empirical_bayes_estimates(df)

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

    # Sorted once and shared, so every array below is in the same order: two
    # sorts of the same frame agree, but only by construction, and `y` lining up
    # with `minutes` is what makes a row a player's match rather than a
    # coincidence.
    ordered = df.sort_values("player_id")

    def per_match(*columns: str) -> np.ndarray:
        """One column per (player, match), or several stacked on a last axis."""
        shape = (
            (nplayer, nmatch, len(columns)) if len(columns) > 1 else (nplayer, nmatch)
        )
        return ordered[list(columns)].to_numpy().reshape(shape)

    return {
        "position": prefix,
        "player_ids": player_ids,
        "nplayer": nplayer,
        "nmatch": nmatch,
        "minutes": per_match("minutes").astype("int64"),
        "y": per_match("goals", "assists", "neither").astype("int64"),
        "alpha": alpha,
        "time_diff": per_match("time_diff"),
        "expected_goals": per_match("expected_goals").astype("float64"),
        "expected_assists": per_match("expected_assists").astype("float64"),
        "team_expected_goals": per_match("team_expected_goals").astype("float64"),
    }
