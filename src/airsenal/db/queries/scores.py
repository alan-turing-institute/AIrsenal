"""Recorded player performances."""

from collections.abc import Sequence
from functools import partial
from typing import overload

import pandas as pd
from sqlalchemy import ColumnElement, and_, func, or_, select
from sqlalchemy.orm import Session

from airsenal.core.logging import get_logger
from airsenal.db.models import Fixture, Player, PlayerAttributes, PlayerScore
from airsenal.db.queries.gameweeks import (
    get_last_complete_gameweek_in_db,
    is_future_gameweek,
)
from airsenal.db.session import get_session
from airsenal.game.scoring import MAX_MINUTES_MATCH
from airsenal.game.season import CURRENT_SEASON

logger = get_logger(__name__)


def get_last_complete_gameweek_of_player_scores_in_db(
    season: str = CURRENT_SEASON, dbsession: Session | None = None
) -> int | None:
    """
    The last gameweek every finished fixture has player scores for.

    The counterpart to `get_last_complete_gameweek_in_db` for the result table. The
    two tables are filled by separate calls that commit separately, so a failure
    between them leaves the player scores behind the results.
    """
    dbsession = get_session(dbsession)
    scored = (
        select(PlayerScore.id)
        .where(PlayerScore.fixture_id == Fixture.fixture_id)
        .exists()
    )
    first_missing = dbsession.scalars(
        select(Fixture)
        .where(
            Fixture.season == season,
            Fixture.gameweek.is_not(None),
            Fixture.result.has(),
            ~scored,
        )
        .order_by(Fixture.gameweek)
        .limit(1)
    ).first()
    if first_missing is not None and first_missing.gameweek is not None:
        return first_missing.gameweek - 1
    # Nothing is missing, so the scores are as far along as the results are.
    return get_last_complete_gameweek_in_db(season=season, dbsession=dbsession)


@overload
def get_player_scores(
    fixture: Fixture, player: Player, dbsession: Session | None = None
) -> PlayerScore | None: ...


@overload
def get_player_scores(
    fixture: Fixture, player: None = None, dbsession: Session | None = None
) -> list[PlayerScore] | None: ...


@overload
def get_player_scores(
    fixture: None = None, *, player: Player, dbsession: Session | None = None
) -> list[PlayerScore] | None: ...


def get_player_scores(
    fixture: Fixture | None = None,
    player: Player | None = None,
    dbsession: Session | None = None,
) -> list[PlayerScore] | PlayerScore | None:
    """
    Player scores for a fixture, for a player, or for one player in one fixture.

    At least one of `fixture` and `player` is required. The return shape follows
    from which: both given returns a single `PlayerScore` (and raises if the
    database holds more than one), either alone returns a list, and no matching
    rows returns None rather than an empty list.
    """
    dbsession = get_session(dbsession)
    if fixture is None and player is None:
        msg = "At least one of fixture and player must be defined"
        raise ValueError(msg)

    query = select(PlayerScore)
    if fixture is not None:
        query = query.where(PlayerScore.fixture_id == fixture.fixture_id)
    if player is not None:
        query = query.where(PlayerScore.player_id == player.player_id)

    player_scores = list(dbsession.scalars(query).all())
    if not player_scores:
        return None

    if fixture is not None and player is not None:
        if len(player_scores) > 1:
            msg = f"More than one score found for player {player} in fixture {fixture}"
            raise ValueError(msg)
        return player_scores[0]
    return player_scores


def get_player_scores_for_gameweeks(
    gameweeks: Sequence[int], season: str, dbsession: Session | None = None
) -> list[PlayerScore]:
    """Every recorded performance in the given gameweeks of a season."""
    dbsession = get_session(dbsession)
    return list(
        dbsession.scalars(
            select(PlayerScore)
            .join(Fixture, PlayerScore.fixture_id == Fixture.fixture_id)
            .where(Fixture.season == season, Fixture.gameweek.in_(list(gameweeks)))
        ).all()
    )


def get_expected_goals_by_fixture(
    dbsession: Session | None = None,
) -> dict[tuple[int, str], float]:
    """
    Each team's expected goals in each fixture, keyed by (fixture id, team).

    Summed over the team's players, because FPL records expected goals per
    player and a team model wants them per side. A fixture with no recorded
    expected goals is absent rather than zero.
    """
    dbsession = get_session(dbsession)
    rows = dbsession.execute(
        select(
            PlayerScore.fixture_id,
            PlayerScore.player_team,
            func.sum(PlayerScore.expected_goals),
        )
        .where(PlayerScore.expected_goals.isnot(None))
        .group_by(PlayerScore.fixture_id, PlayerScore.player_team)
    ).all()
    return {
        (int(fixture_id), str(team)): float(total)
        for fixture_id, team, total in rows
        if total is not None
    }


def was_available() -> ColumnElement[bool]:
    """
    Whether a score is from a match the player was available for.

    Played at least 60 minutes, or was not flagged: a 100% chance of playing, or
    no availability recorded for the match.
    """
    return or_(
        PlayerScore.minutes >= 60,
        PlayerScore.chance_of_playing == 100,
        PlayerScore.chance_of_playing.is_(None),
    )


def get_recent_playerscore_rows(
    player: Player,
    n_matches_to_use: int,
    season: str,
    last_gameweek: int,
    exclude_unavailable: bool = False,
    current_team_only: bool = False,
    dbsession: Session | None = None,
) -> list[PlayerScore]:
    """This player's last `n_matches_to_use` scores, most recent first."""
    dbsession = get_session(dbsession)
    # If asking for gameweeks without results in DB, revert to most recent results.
    last_available_gameweek = get_last_complete_gameweek_in_db(
        season=season, dbsession=dbsession
    )
    if not last_available_gameweek:
        # e.g. before this season has started
        return []

    last_gameweek = min(last_gameweek, last_available_gameweek)

    # get the playerscore rows from the db
    query = (
        select(PlayerScore)
        .join(Fixture, PlayerScore.fixture_id == Fixture.fixture_id)
        .where(
            Fixture.season == season,
            PlayerScore.player_id == player.player_id,
            Fixture.gameweek <= last_gameweek,
        )
    )
    if exclude_unavailable:
        query = query.where(was_available())
    if current_team_only:
        team = player.team(last_gameweek, season)
        query = query.where(PlayerScore.player_team == team)

    return list(
        dbsession.scalars(
            query.order_by(Fixture.gameweek.desc()).limit(n_matches_to_use)
        ).all()
    )


def get_playerscores_for_player_gameweek(
    player_id: int | str,
    gameweek: int,
    season: str = CURRENT_SEASON,
    dbsession: Session | None = None,
) -> list[PlayerScore]:
    """This player's scores in a gameweek - more than one if it is a double."""
    dbsession = get_session(dbsession)
    return list(
        dbsession.scalars(
            select(PlayerScore)
            .join(Fixture, PlayerScore.fixture_id == Fixture.fixture_id)
            .where(
                Fixture.season == season,
                PlayerScore.player_id == player_id,
                Fixture.gameweek == gameweek,
            )
        ).all()
    )


def get_player_scores_df(
    *,
    min_minutes: int = 0,
    max_minutes: int = MAX_MINUTES_MATCH,
    position: str | None = None,
    gameweek: int,
    season: str,
    dbsession: Session | None = None,
) -> pd.DataFrame:
    """Player scores, filtered by minutes played and position."""
    dbsession = get_session(dbsession)
    query = (
        select(PlayerScore, Fixture.season, Fixture.gameweek, PlayerAttributes.position)
        .where(PlayerScore.minutes >= min_minutes)
        .where(PlayerScore.minutes <= max_minutes)
        .join(Fixture)
        .join(
            PlayerAttributes,
            and_(
                PlayerAttributes.player_id == PlayerScore.player_id,
                PlayerAttributes.season == Fixture.season,
                PlayerAttributes.gameweek == Fixture.gameweek,
            ),
        )
        .order_by(Fixture.season, Fixture.gameweek, PlayerAttributes.player_id)
    )
    if position:
        query = query.where(PlayerAttributes.position == position)

    df = pd.read_sql(query, dbsession.connection())

    is_fut = partial(
        is_future_gameweek, current_season=season, current_gameweek=gameweek
    )
    exclude = df.apply(lambda r: is_fut(r["gameweek"], r["season"]), axis=1)
    return df[~exclude]
