"""
Filling the player prediction table.

Each run gets a "tag" of its own, stored alongside the rows it writes, so that
an optimizer asked for a tag reads one consistent set of predictions rather than
a mix of two runs.
"""

from uuid import uuid4

from sqlalchemy.orm.session import Session

from airsenal.core.console import console, track
from airsenal.core.logging import get_logger
from airsenal.db.queries.fixtures import get_fixtures_for_gameweeks
from airsenal.db.queries.players import list_players
from airsenal.db.session import get_session
from airsenal.game.scoring import MAX_GOALS
from airsenal.game.season import CURRENT_SEASON
from airsenal.prediction.features import (
    fit_bonus_points,
    fit_card_points,
    fit_def_con,
    fit_save_points,
)
from airsenal.prediction.player_models.fitting import get_all_fitted_player_data
from airsenal.prediction.points import PointsConfig, calc_predicted_points_for_player
from airsenal.prediction.protocols import PlayerModel, TeamModel
from airsenal.prediction.team_models import (
    build_team_model,
)
from airsenal.prediction.team_models.fitting import (
    get_fitted_team_model,
    get_goal_probabilities_for_fixtures,
)

logger = get_logger(__name__)


def calc_all_predicted_points(
    gameweeks: list[int],
    season: str,
    dbsession: Session,
    points: PointsConfig | None = None,
    tag: str = "",
    player_model: PlayerModel | None = None,
    team_model: TeamModel | None = None,
) -> None:
    """
    Do the full prediction for players.
    """
    points = points if points is not None else PointsConfig()
    model_team = get_fitted_team_model(
        season=season,
        gameweek=min(gameweeks),
        dbsession=dbsession,
        model=team_model if team_model is not None else build_team_model(),
    )
    logger.info("Calculating fixture score probabilities...")
    fixtures = get_fixtures_for_gameweeks(gameweeks, season=season, dbsession=dbsession)
    fixture_goal_probs = get_goal_probabilities_for_fixtures(
        fixtures, model_team, max_goals=MAX_GOALS
    )

    df_player = get_all_fitted_player_data(
        season, gameweeks[0], model=player_model, dbsession=dbsession
    )

    df_bonus = fit_bonus_points(gameweeks[0], season) if points.bonus else None
    df_saves = fit_save_points(gameweeks[0], season) if points.saves else None
    df_cards = fit_card_points(gameweeks[0], season) if points.cards else None
    df_def_con = fit_def_con(gameweeks[0], season) if points.def_con else None

    players = list_players(season=season, gameweek=gameweeks[0], dbsession=dbsession)

    for player in track(players, description="Predicting player points:"):
        predictions = calc_predicted_points_for_player(
            player,
            fixture_goal_probs,
            df_player,
            df_bonus,
            df_saves,
            df_cards,
            df_def_con,
            season,
            gameweeks=gameweeks,
            tag=tag,
            dbsession=dbsession,
        )
        for pred in predictions:
            dbsession.add(pred)
    dbsession.commit()
    logger.info("Finished adding predictions to db")


def make_predictedscore_table(
    gameweeks: list[int],
    season: str = CURRENT_SEASON,
    points: PointsConfig | None = None,
    tag_prefix: str | None = None,
    player_model: PlayerModel | None = None,
    team_model: TeamModel | None = None,
    dbsession: Session | None = None,
) -> str:
    """
    Predict every player's points over `gameweeks`, and return the tag written.

    `gameweeks` is required: this used to default to three weeks from the next
    one, a second hardcoded window alongside the one `get_gameweeks_array` had,
    and the two could disagree. Resolving a window is
    `AIrsenalPipeline.gameweeks`' job, and every caller goes through it.
    """
    dbsession = dbsession if dbsession is not None else get_session()
    tag = tag_prefix or ""
    tag += str(uuid4())
    with console.status("Predicting points..."):
        calc_all_predicted_points(
            gameweeks=gameweeks,
            season=season,
            dbsession=dbsession,
            points=points,
            tag=tag,
            player_model=player_model,
            team_model=team_model,
        )
    return tag
