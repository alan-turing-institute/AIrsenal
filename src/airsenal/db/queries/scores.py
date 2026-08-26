"""Recorded player performances."""

from functools import partial

import pandas as pd
from sqlalchemy import and_, or_, select
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
    dbsession = dbsession if dbsession is not None else get_session()
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


def get_recent_playerscore_rows(
    player: Player,
    n_matches_to_use: int = 3,
    season: str = CURRENT_SEASON,
    last_gw: int | None = None,
    exclude_unavailable: bool = False,
    current_team_only: bool = False,
    dbsession: Session | None = None,
) -> list[PlayerScore]:
    """This player's last `n_matches_to_use` scores, most recent first."""
    dbsession = dbsession if dbsession is not None else get_session()
    # If asking for gameweeks without results in DB, revert to most recent results.
    last_available_gameweek = get_last_complete_gameweek_in_db(
        season=season, dbsession=dbsession
    )
    if not last_available_gameweek:
        # e.g. before this season has started
        return []

    if last_gw is None and season != CURRENT_SEASON:
        msg = "last_gw must be specified is running on previous seasons"
        raise ValueError(msg)

    if last_gw is None or last_gw > last_available_gameweek:
        last_gw = last_available_gameweek

    # get the playerscore rows from the db
    query = (
        select(PlayerScore)
        .join(Fixture, PlayerScore.fixture_id == Fixture.fixture_id)
        .where(
            Fixture.season == season,
            PlayerScore.player_id == player.player_id,
            Fixture.gameweek <= last_gw,
        )
    )
    if exclude_unavailable:
        # minutes at least 60 or no flag status (100% chance of playing)
        query = query.where(
            or_(
                PlayerScore.minutes >= 60,
                PlayerScore.chance_of_playing == 100,
                # rows written before chance_of_playing was recorded
                PlayerScore.chance_of_playing.is_(None),
            )
        )
    if current_team_only:
        team = player.team(season, last_gw)
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
    dbsession = dbsession if dbsession is not None else get_session()
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
    season: str,
    gameweek: int,
    min_minutes: int = 0,
    max_minutes: int = MAX_MINUTES_MATCH,
    position: str | None = None,
    dbsession: Session | None = None,
) -> pd.DataFrame:
    """Player scores, filtered by minutes played and position."""
    dbsession = dbsession if dbsession is not None else get_session()
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

    is_fut = partial(is_future_gameweek, current_season=season, next_gameweek=gameweek)
    exclude = df.apply(lambda r: is_fut(r["season"], r["gameweek"]), axis=1)
    return df[~exclude]
