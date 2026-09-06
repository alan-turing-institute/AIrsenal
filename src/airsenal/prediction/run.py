"""Filling the player prediction table."""

from uuid import uuid4

from sqlalchemy.orm.session import Session

from airsenal.core.console import console, track
from airsenal.core.logging import get_logger
from airsenal.db.models import Fixture, Player, PlayerPrediction
from airsenal.db.queries.fixtures import get_fixtures_for_player
from airsenal.db.queries.players import list_players
from airsenal.db.session import get_session
from airsenal.game.season import CURRENT_SEASON
from airsenal.prediction.points_models import build_points_model
from airsenal.prediction.protocols import (
    PointsFitRequest,
    PointsModel,
    PointsRequest,
)

logger = get_logger(__name__)


def make_prediction(
    player: Player, fixture: Fixture, points: float, tag: str
) -> PlayerPrediction:
    """Instantiate and populate a PlayerPrediction schema object."""
    pp = PlayerPrediction()
    pp.predicted_points = points
    pp.tag = tag
    pp.player = player
    pp.fixture = fixture
    return pp


def calc_all_predicted_points(
    gameweeks: list[int],
    *,
    tag: str = "",
    season: str,
    dbsession: Session,
    points_model: PointsModel | None = None,
) -> None:
    """Predict every player's points for the given gameweeks, and write them out."""
    model = points_model if points_model is not None else build_points_model()
    # Everything is fitted as at the first gameweek of the window - the one we
    # are predicting *from* - which is also what decides which players there are
    # to predict for.
    root_gameweek = min(gameweeks)
    model = model.fit(
        PointsFitRequest(gameweeks=gameweeks, season=season, dbsession=dbsession)
    )

    players = list_players(season=season, gameweek=root_gameweek, dbsession=dbsession)

    for player in track(players, description="Predicting player points:"):
        for fixture in get_fixtures_for_player(
            player, season, gameweeks=gameweeks, dbsession=dbsession
        ):
            if fixture.gameweek is None:
                logger.warning("Skipping fixture %s with no gameweek", fixture)
                continue
            prediction = model.predict(
                PointsRequest(
                    player=player,
                    fixture=fixture,
                    root_gameweek=root_gameweek,
                    season=season,
                    dbsession=dbsession,
                )
            )
            dbsession.add(
                make_prediction(player, fixture, prediction.expected_points, tag)
            )
    dbsession.commit()
    logger.info("Finished adding predictions to db")


def make_predictedscore_table(
    gameweeks: list[int],
    season: str = CURRENT_SEASON,
    tag_prefix: str | None = None,
    points_model: PointsModel | None = None,
    dbsession: Session | None = None,
) -> str:
    """Predict every player's points over `gameweeks`, and return the tag written."""
    dbsession = dbsession if dbsession is not None else get_session()
    tag = tag_prefix or ""
    tag += str(uuid4())
    with console.status("Predicting points..."):
        calc_all_predicted_points(
            gameweeks=gameweeks,
            season=season,
            dbsession=dbsession,
            tag=tag,
            points_model=points_model,
        )
    return tag
