"""Estimating how many minutes a player will play."""

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from airsenal.db.models import Fixture, Player, PlayerScore
from airsenal.db.queries.gameweeks import next_gameweek
from airsenal.db.queries.scores import get_recent_playerscore_rows
from airsenal.db.session import get_session
from airsenal.game.season import CURRENT_SEASON, get_previous_season


def calc_average_minutes(player_scores: list[PlayerScore]) -> float:
    """Mean minutes played across a list of PlayerScore rows."""
    total = 0.0
    for ps in player_scores:
        total += ps.minutes
    return total / len(player_scores)


def estimate_minutes_from_prev_season(
    player: Player,
    gameweek: int | None = None,
    season: str = CURRENT_SEASON,
    n_matches_to_use: int = 10,
    exclude_unavailable: bool = True,
    current_team_only: bool = True,
    dbsession: Session | None = None,
) -> list[float]:
    """Mean minutes in the previous season, or [0] if we have none."""
    gameweek = next_gameweek() if gameweek is None else gameweek
    dbsession = dbsession if dbsession is not None else get_session()
    previous_season = get_previous_season(season)

    # Only consider minutes the player played with his current team
    current_team = player.team(gameweek, season)
    query = (
        select(PlayerScore)
        .join(Fixture, PlayerScore.fixture)
        .where(
            PlayerScore.player_id == player.player_id,
            Fixture.season == previous_season,
        )
    )

    if current_team_only:
        current_team = player.team(gameweek, season)
        query = query.where(PlayerScore.player_team == current_team)

    if exclude_unavailable:
        query = query.where(
            or_(
                PlayerScore.minutes >= 60,
                PlayerScore.chance_of_playing == 100,
                # rows written before chance_of_playing was recorded
                PlayerScore.chance_of_playing.is_(None),
            )
        )

    player_scores = list(
        dbsession.scalars(
            query.order_by(Fixture.gameweek.desc()).limit(n_matches_to_use)
        ).all()
    )

    if len(player_scores) == 0:
        # no FPL history / didn't play for current team last season
        return [0]

    # Return average minutes. A weakness of this is increased rotation at the end of the
    # season when teams don't have anything to play for.
    return [calc_average_minutes(player_scores)]


def get_recent_minutes_for_player(
    player: Player,
    n_matches_to_use: int = 3,
    season: str = CURRENT_SEASON,
    last_gw: int | None = None,
    exclude_unavailable: bool = True,
    current_team_only: bool = True,
    dbsession: Session | None = None,
) -> list[float]:
    """
    Minutes played in each of the last `n_matches_to_use` matches.

    `current_gw` defaults to the most recent finished gameweek.
    """
    dbsession = dbsession if dbsession is not None else get_session()
    if last_gw is None:
        if season != CURRENT_SEASON:
            msg = "last_gw must be defined if running on previous seasons"
            raise ValueError(msg)
        last_gw = next_gameweek()

    playerscores = (
        get_recent_playerscore_rows(
            player,
            n_matches_to_use,
            season,
            last_gw,
            exclude_unavailable,
            current_team_only,
            dbsession,
        )
        or []
    )

    minutes = [float(r.minutes) for r in playerscores]

    if len(minutes) < n_matches_to_use:
        minutes += estimate_minutes_from_prev_season(
            player, gameweek=last_gw, season=season, dbsession=dbsession
        )
    return minutes or [0.0]
