"""
A small but complete database, for exercising the whole pipeline.

Built in code rather than loaded from CSV so that there is no second data format
to keep in step with the schema - the builder fails to compile if a column is
renamed, where a CSV would just load the wrong thing.

Deliberately tiny: eight teams and enough players to fill a squad twice over,
two past seasons of results to fit on and three future gameweeks to predict.
Everything is derived from the indices, so the whole database is reproducible
and there is no RNG to seed.
"""

from datetime import date, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from airsenal.core.enums import Position
from airsenal.db.models import (
    Base,
    FifaTeamRating,
    Fixture,
    Player,
    PlayerAttributes,
    PlayerScore,
    Result,
    Team,
)
from airsenal.db.session import configure_database

SEASON = "2526"
PAST_SEASONS = ["2324", "2425"]
TEAMS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH"]
GAMEWEEKS_PER_PAST_SEASON = 8
FUTURE_GAMEWEEKS = [1, 2, 3]

# Enough of each position to build a legal squad (2/5/5/3) with room to transfer,
# while staying under three players per team.
SQUAD_SHAPE = {Position.GK: 8, Position.DEF: 16, Position.MID: 16, Position.FWD: 8}

BASE_PRICE = {Position.GK: 40, Position.DEF: 40, Position.MID: 50, Position.FWD: 60}
PRICE_STEP = {Position.GK: 2, Position.DEF: 3, Position.MID: 5, Position.FWD: 6}

FIRST_DATE = date(2024, 8, 10)


def _players() -> list[tuple[int, str, Position, str, int]]:
    """(player_id, name, position, team, price) for every player."""
    rows = []
    player_id = 0
    for position, count in SQUAD_SHAPE.items():
        for i in range(count):
            team = TEAMS[i % len(TEAMS)]
            # tiered prices, so the optimiser has both cheap and expensive options
            price = BASE_PRICE[position] + PRICE_STEP[position] * (i % 5)
            rows.append((player_id, f"{position}_{i}", position, team, price))
            player_id += 1
    return rows


def _round_robin(gameweek: int) -> list[tuple[str, str]]:
    """Four fixtures pairing all eight teams, rotated by gameweek."""
    rotated = [
        TEAMS[0],
        *TEAMS[1:][(gameweek - 1) % 7 :],
        *TEAMS[1:][: (gameweek - 1) % 7],
    ]
    half = len(rotated) // 2
    return list(zip(rotated[:half], reversed(rotated[half:]), strict=True))


def _build(session: Session) -> None:
    for season in [*PAST_SEASONS, SEASON]:
        for i, name in enumerate(TEAMS):
            session.add(
                Team(name=name, full_name=f"{name} FC", season=season, team_id=i + 1)
            )
            # the team model uses these as covariates, so every season needs them
            session.add(
                FifaTeamRating(
                    season=season,
                    team=name,
                    att=70 + i,
                    defn=70 + (i + 3) % 8,
                    mid=70 + (i + 5) % 8,
                    ovr=70 + (i + 1) % 8,
                )
            )

    players = _players()
    for player_id, name, _position, _team, _price in players:
        session.add(Player(player_id=player_id, fpl_api_id=player_id, name=name))

    for season in [*PAST_SEASONS, SEASON]:
        gameweeks = (
            range(1, GAMEWEEKS_PER_PAST_SEASON + 1)
            if season != SEASON
            else FUTURE_GAMEWEEKS
        )
        for player_id, _name, position, team, price in players:
            for gameweek in gameweeks:
                session.add(
                    PlayerAttributes(
                        player_id=player_id,
                        season=season,
                        gameweek=gameweek,
                        price=price,
                        team=team,
                        position=str(position),
                    )
                )

    fixture_id = 0
    result_id = 0
    match_day = FIRST_DATE
    for season in [*PAST_SEASONS, SEASON]:
        played = season != SEASON
        gameweeks = (
            range(1, GAMEWEEKS_PER_PAST_SEASON + 1) if played else FUTURE_GAMEWEEKS
        )
        for gameweek in gameweeks:
            for home, away in _round_robin(gameweek):
                fixture_id += 1
                match_day += timedelta(days=1)
                fixture = Fixture(
                    fixture_id=fixture_id,
                    date=match_day.isoformat(),
                    gameweek=gameweek,
                    home_team=home,
                    away_team=away,
                    season=season,
                    tag="e2e",
                )
                session.add(fixture)
                if not played:
                    continue

                result_id += 1
                # scorelines vary with the gameweek so the team model has
                # something to fit, but stay small and deterministic
                home_score = (gameweek + fixture_id) % 4
                away_score = (gameweek + fixture_id) % 3
                session.add(
                    Result(
                        result_id=result_id,
                        fixture_id=fixture_id,
                        home_score=home_score,
                        away_score=away_score,
                    )
                )
                _add_player_scores(
                    session,
                    players,
                    fixture,
                    result_id,
                    home,
                    away,
                    home_score,
                    away_score,
                )
    session.commit()


def _add_player_scores(
    session: Session,
    players: list[tuple[int, str, Position, str, int]],
    fixture: Fixture,
    result_id: int,
    home: str,
    away: str,
    home_score: int,
    away_score: int,
) -> None:
    for player_id, _name, position, team, _price in players:
        if team not in (home, away):
            continue
        opponent = away if team == home else home
        scored = home_score if team == home else away_score
        conceded = away_score if team == home else home_score
        # a fixed share of the team's goals, so the player model has signal
        goals = scored if position is Position.FWD and player_id % 3 == 0 else 0
        assists = scored if position is Position.MID and player_id % 4 == 0 else 0
        session.add(
            PlayerScore(
                player_id=player_id,
                result_id=result_id,
                fixture_id=fixture.fixture_id,
                player_team=team,
                opponent=opponent,
                minutes=90 if player_id % 5 else 45,
                goals=goals,
                assists=assists,
                conceded=conceded,
                bonus=1 if goals else 0,
                points=2 + 4 * goals + 3 * assists,
                clean_sheets=int(conceded == 0),
                bps=10 + goals,
                saves=0,
                yellow_cards=0,
                red_cards=0,
                own_goals=0,
                penalties_saved=0,
                penalties_missed=0,
            )
        )


@pytest.fixture(scope="session")
def pipeline_db(tmp_path_factory):
    """
    A complete, tiny database, built once for the whole e2e module.

    The package default session is pointed here too, so that code which has not
    (yet) had a dbsession threaded through it reads this database rather than
    whatever conftest set up for the unit tests.
    """
    path = tmp_path_factory.mktemp("e2e") / "pipeline.db"
    connection_string = f"sqlite:///{path}"
    engine = create_engine(connection_string)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    with factory() as session:
        _build(session)

    configure_database(connection_string)
    try:
        with factory() as session:
            yield session
    finally:
        configure_database(None)
