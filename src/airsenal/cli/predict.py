"""Commands for predicting player scores."""

from airsenal.cli import options
from airsenal.db.session import session_scope
from airsenal.game.season import CURRENT_SEASON
from airsenal.pipeline import AIrsenalPipeline, PipelineSettings
from airsenal.pipeline.settings import DEFAULT_N_GAMEWEEKS
from airsenal.prediction.minutes_models import DEFAULT_MINUTES_MODEL
from airsenal.prediction.player_models import DEFAULT_PLAYER_MODEL
from airsenal.prediction.point_components import PointsConfig
from airsenal.prediction.points_models import (
    DEFAULT_POINTS_MODEL,
    build_points_model,
)
from airsenal.prediction.team_models import DEFAULT_TEAM_MODEL
from airsenal.reporting.top_players import get_top_predicted_points


def predict(
    n_gameweeks: options.OptionalNGameweeks = None,
    gameweek_start: options.GameweekStart = None,
    gameweek_end: options.GameweekEnd = None,
    season: options.Season = CURRENT_SEASON,
    bonus: options.Bonus = True,
    cards: options.Cards = True,
    saves: options.Saves = True,
    def_con: options.DefCon = True,
    player_model: options.PlayerModel = DEFAULT_PLAYER_MODEL,
    team_model: options.TeamModel = DEFAULT_TEAM_MODEL,
    minutes_model: options.MinutesModel = DEFAULT_MINUTES_MODEL,
    points_model: options.PointsModel = DEFAULT_POINTS_MODEL,
    epsilon: options.Epsilon = None,
) -> None:
    """Predict player scores for a gameweek range."""
    pipeline = AIrsenalPipeline(
        points_model=build_points_model(
            points_model,
            team_model=team_model,
            player_model=player_model,
            minutes_model=minutes_model,
            epsilon=epsilon,
            points=PointsConfig(bonus=bonus, cards=cards, saves=saves, def_con=def_con),
        ),
        settings=PipelineSettings(
            season=season,
            n_gameweeks=n_gameweeks or DEFAULT_N_GAMEWEEKS,
            gameweek_start=gameweek_start,
            gameweek_end=gameweek_end,
            refresh_database=False,
        ),
    )
    with session_scope() as session:
        session.expire_on_commit = False
        gameweeks = pipeline.gameweeks(session)
        tag = pipeline.predict(gameweeks, session)
        get_top_predicted_points(
            gameweeks=gameweeks,
            tag=tag,
            season=season,
            per_position=True,
            n_players=5,
            dbsession=session,
        )
