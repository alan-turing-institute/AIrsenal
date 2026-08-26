"""Fixture lookups."""

from collections.abc import Iterable
from datetime import date, datetime

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from airsenal.core.dates import parse_date
from airsenal.core.logging import get_logger
from airsenal.db.models import Fixture, Player
from airsenal.db.queries.gameweeks import next_gameweek
from airsenal.db.queries.tags import get_latest_fixture_tag
from airsenal.db.queries.teams import get_team_name
from airsenal.db.session import get_session
from airsenal.game.season import CURRENT_SEASON

logger = get_logger(__name__)


def get_fixtures_for_player(
    player: Player | str | int,
    season: str = CURRENT_SEASON,
    gameweeks: list[int] | None = None,
    dbsession: Session | None = None,
) -> list[Fixture]:
    """
    A player's upcoming fixtures, by player id or name.

    Without `gameweeks`: the rest of the season for the current one, and the
    whole season for a past one.
    """
    dbsession = dbsession if dbsession is not None else get_session()
    if isinstance(player, str):  # given a player name
        player_record = dbsession.scalars(
            select(Player).where(Player.name == player).limit(1)
        ).first()
    elif isinstance(player, int):  # given a player id
        player_record = dbsession.scalars(
            select(Player).where(Player.player_id == player).limit(1)
        ).first()
    else:  # given a player object
        player_record = player
    if not player_record:
        logger.warning("Couldn't find %s in database", player)
        return []
    if not gameweeks and season != CURRENT_SEASON:
        msg = "Gameweek range must be specified for past seasons"
        raise ValueError(msg)
    if not gameweeks:
        team = player_record.team(season, next_gameweek())
    else:
        team = player_record.team(season, gameweeks[0])  # same team for whole gameweeks
    tag = get_latest_fixture_tag(season, dbsession)
    fixture_rows = dbsession.scalars(
        select(Fixture)
        .where(
            Fixture.season == season,
            Fixture.tag == tag,
            or_(Fixture.home_team == team, Fixture.away_team == team),
        )
        .order_by(Fixture.gameweek)
    ).all()
    fixtures = []
    for fixture in fixture_rows:
        if not fixture.gameweek:  # fixture not scheduled yet
            continue
        if gameweeks:
            if fixture.gameweek in gameweeks:
                fixtures.append(fixture)
        else:
            if season == CURRENT_SEASON and fixture.gameweek < next_gameweek():
                continue
            logger.debug("%s", fixture)
            fixtures.append(fixture)
    return fixtures


def get_fixtures_for_season(
    season: str = CURRENT_SEASON, dbsession: Session | None = None
) -> list[Fixture]:
    """Every fixture in a season."""
    dbsession = dbsession if dbsession is not None else get_session()
    return list(
        dbsession.scalars(select(Fixture).where(Fixture.season == season)).all()
    )


def get_fixtures_for_gameweeks(
    gameweeks: Iterable[int],
    season: str = CURRENT_SEASON,
    dbsession: Session | None = None,
) -> list[Fixture]:
    """
    Get a list of fixtures for the specified gameweeks.

    Callers with a single gameweek pass `[gameweek]`.
    """
    dbsession = dbsession if dbsession is not None else get_session()
    return list(
        dbsession.scalars(
            select(Fixture).where(
                Fixture.season == season, Fixture.gameweek.in_(list(gameweeks))
            )
        ).all()
    )


def get_fixture_teams(fixtures: Iterable[Fixture]) -> list[tuple[str, str]]:
    """(home_team, away_team) for each of these fixtures."""
    return [(fixture.home_team, fixture.away_team) for fixture in fixtures]


def find_fixture(
    team: str | int,
    was_home: bool | None = None,
    other_team: str | int | None = None,
    gameweek: int | None = None,
    season: str = CURRENT_SEASON,
    kickoff_time: date | datetime | str | None = None,
    dbsession: Session | None = None,
    verbose: bool = True,
) -> Fixture | None:
    """
    The one fixture matching a team and any of the other filters given.

    Raises:
        ValueError: The filters match no fixture, or more than one.
    """
    dbsession = dbsession if dbsession is not None else get_session()
    if not isinstance(team, str):
        team_name = get_team_name(team, season=season, dbsession=dbsession)
    else:
        team_name = team

    if not team_name:
        msg = f"No team with id {team} in {season} season"
        raise ValueError(msg)

    if isinstance(other_team, int):
        other_team_name = get_team_name(other_team, season=season, dbsession=dbsession)
    else:
        other_team_name = other_team

    query = select(Fixture).where(Fixture.season == season)
    if gameweek:
        query = query.where(Fixture.gameweek == gameweek)
    if was_home is True:
        query = query.where(Fixture.home_team == team_name)
    elif was_home is False:
        query = query.where(Fixture.away_team == team_name)
    else:
        query = query.where(
            or_(Fixture.away_team == team_name, Fixture.home_team == team_name)
        )

    if other_team_name:
        if was_home is True:
            query = query.where(Fixture.away_team == other_team_name)
        elif was_home is False:
            query = query.where(Fixture.home_team == other_team_name)
        elif was_home is None:
            query = query.where(
                or_(
                    Fixture.away_team == other_team_name,
                    Fixture.home_team == other_team_name,
                )
            )

    fixtures = dbsession.scalars(query).all()

    if not fixtures or len(fixtures) == 0:
        if verbose:
            logger.warning(
                "No fixture with season=%s, gw=%s, team_name=%s, was_home=%s, "
                "other_team_name=%s, kickoff_time=%s",
                season,
                gameweek,
                team_name,
                was_home,
                other_team_name,
                kickoff_time,
            )
        return None

    if len(fixtures) == 1:
        return fixtures[0]
    if kickoff_time:
        # team played multiple games in the gameweek, determine the
        # fixture of interest using the kickoff time,
        kickoff_date = parse_date(kickoff_time)

        for f in fixtures:
            f_date = parse_date(f.date)
            if f_date == kickoff_date:
                return f

    logger.warning(
        "No unique fixture with season=%s, gw=%s, team_name=%s, was_home=%s, "
        "kickoff_time=%s",
        season,
        gameweek,
        team_name,
        was_home,
        kickoff_time,
    )
    return None


def get_player_team_from_fixture(
    fixture: Fixture,
    opponent: str | int | None = None,
    player_at_home: bool | None = None,
    season: str = CURRENT_SEASON,
    dbsession: Session | None = None,
) -> str:
    """
    The team a player turned out for, identified by gameweek, opponent and venue.

    With `return_fixture`, returns (team_name, fixture) rather than just the name.
    """
    dbsession = dbsession if dbsession is not None else get_session()
    if opponent is None and player_at_home is None:
        msg = "Either opponent or player_at_home must be specified"
        raise ValueError(msg)

    if player_at_home is not None:
        return fixture.home_team if player_at_home else fixture.away_team

    if isinstance(opponent, int):
        opponent_name = get_team_name(opponent, season=season, dbsession=dbsession)
    else:
        opponent_name = opponent

    if fixture.home_team == opponent_name:
        return fixture.away_team
    if fixture.away_team == opponent_name:
        return fixture.home_team

    msg = f"Opponent {opponent_name} not in fixture"
    raise ValueError(msg)
