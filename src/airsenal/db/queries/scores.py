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
    next_gameweek,
)
from airsenal.db.session import get_session
from airsenal.domain.scoring import MAX_MINUTES_MATCH
from airsenal.domain.season import CURRENT_SEASON

logger = get_logger(__name__)


def get_player_scores(
    fixture: Fixture | None = None,
    player: Player | None = None,
    dbsession: Session | None = None,
) -> list[PlayerScore] | PlayerScore | None:
    """
    Get player scores for a fixture.
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


def get_previous_points_for_same_fixture(
    player: str | int, fixture_id: int, dbsession: Session | None = None
) -> dict[str, int]:
    """
    Search the past matches for same fixture in past seasons,
    and how many points the player got.
    """
    dbsession = dbsession if dbsession is not None else get_session()
    if isinstance(player, str):
        player_record = dbsession.scalars(
            select(Player).where(Player.name == player).limit(1)
        ).first()
        if not player_record:
            logger.warning("Can't find player %s", player)
            return {}
        player_id = player_record.player_id
    else:
        player_id = player
    fixture = dbsession.scalars(
        select(Fixture).where(Fixture.fixture_id == fixture_id).limit(1)
    ).first()
    if not fixture:
        logger.warning("Couldn't find fixture_id %s", fixture_id)
        return {}
    home_team = fixture.home_team
    away_team = fixture.away_team

    previous_matches = dbsession.scalars(
        select(Fixture)
        .where(Fixture.home_team == home_team, Fixture.away_team == away_team)
        .order_by(Fixture.season)
    ).all()
    fixture_seasons = {f.fixture_id: f.season for f in previous_matches}
    if not fixture_seasons:
        return {}

    previous_points = {}
    scores = dbsession.scalars(
        select(PlayerScore).where(
            PlayerScore.player_id == player_id,
            PlayerScore.fixture_id.in_(fixture_seasons.keys()),
        )
    ).all()
    for score in scores:
        season = fixture_seasons.get(score.fixture_id)
        if season is not None:
            previous_points[season] = score.points

    return previous_points


def get_recent_playerscore_rows(
    player: Player,
    n_matches_to_use: int = 3,
    season: str = CURRENT_SEASON,
    last_gw: int | None = None,
    exclude_unavailable: bool = False,
    current_team_only: bool = False,
    dbsession: Session | None = None,
) -> list[PlayerScore]:
    """
    Query the playerscore table in the database to retrieve
    the last n_matches_to_use rows for this player.
    """
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
                PlayerScore.chance_of_playing.is_(None),  # for backwards compatibility
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
    player: Player,
    gameweek: int,
    season: str = CURRENT_SEASON,
    dbsession: Session | None = None,
) -> list[PlayerScore]:
    """
    FPL points for this player for selected match.
    Returns a PlayerScore object.
    """
    dbsession = dbsession if dbsession is not None else get_session()
    return list(
        dbsession.scalars(
            select(PlayerScore)
            .join(Fixture, PlayerScore.fixture_id == Fixture.fixture_id)
            .where(
                Fixture.season == season,
                PlayerScore.player_id == player.player_id,
                Fixture.gameweek == gameweek,
            )
        ).all()
    )


def get_recent_scores_for_player(
    player: Player,
    n_matches_to_use: int = 3,
    season: str = CURRENT_SEASON,
    last_gw: int | None = None,
    exclude_unavailable: bool = False,
    current_team_only: bool = False,
    dbsession: Session | None = None,
) -> dict[int, int]:
    """
    Look n_matches_to_use matches back, and return the
    FPL points for this player for each of these matches.
    Return a dict {gameweek: score, }
    """
    dbsession = dbsession if dbsession is not None else get_session()
    if last_gw is None:
        if season != CURRENT_SEASON:
            msg = "last_gw must be specified if running on previous seasons"
            raise ValueError(msg)
        last_gw = next_gameweek()
    first_gw = last_gw - n_matches_to_use

    playerscores = get_recent_playerscore_rows(
        player,
        n_matches_to_use,
        season,
        last_gw,
        exclude_unavailable,
        current_team_only,
        dbsession,
    )
    if not playerscores:  # e.g. start of season
        return {}

    return {range(first_gw, last_gw)[i]: ps.points for i, ps in enumerate(playerscores)}


def get_player_scores_df(
    season: str,
    gameweek: int,
    min_minutes: int = 0,
    max_minutes: int = MAX_MINUTES_MATCH,
    position: str | None = None,
    dbsession: Session | None = None,
) -> pd.DataFrame:
    """
    Query player scores filtered by played minutes and position.
    """
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
