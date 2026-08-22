"""
Fill the "player_prediction" table with score predictions
Usage:
python fill_predictedscore_table.py --n_gameweeks <nweeks>
Generates a "tag" string which is stored so it can later be used by team-optimizers to
get consistent sets of predictions from the database.
"""

from uuid import uuid4

from sqlalchemy.orm.session import Session

from airsenal.core.console import console, track
from airsenal.core.logging import get_logger
from airsenal.db.queries.fixtures import get_fixtures_for_gameweeks
from airsenal.db.queries.gameweeks import get_gameweeks_array, next_gameweek
from airsenal.db.queries.players import list_players
from airsenal.db.session import get_session, session_scope
from airsenal.domain.scoring import MAX_GOALS
from airsenal.domain.season import CURRENT_SEASON
from airsenal.prediction.features import (
    fit_bonus_points,
    fit_card_points,
    fit_def_con,
    fit_save_points,
    get_all_fitted_player_data,
)
from airsenal.prediction.points import calc_predicted_points_for_player
from airsenal.prediction.protocols import PlayerModel, TeamModel
from airsenal.prediction.registry import PLAYER_MODELS, TEAM_MODELS
from airsenal.prediction.team_models.dixon_coles import (
    DEFAULT_TEAM_EPSILON,
    get_fitted_team_model,
    get_goal_probabilities_for_fixtures,
)
from airsenal.reporting.top_players import get_top_predicted_points

logger = get_logger(__name__)


def calc_all_predicted_points(
    gameweeks: list[int],
    season: str,
    dbsession: Session,
    include_bonus: bool = True,
    include_cards: bool = True,
    include_saves: bool = True,
    include_def_con: bool = True,
    tag: str = "",
    player_model: PlayerModel | None = None,
    team_model: TeamModel | None = None,
    team_model_args: dict | None = None,
) -> None:
    """
    Do the full prediction for players.
    """
    if team_model_args is None:
        team_model_args = {"epsilon": DEFAULT_TEAM_EPSILON}
    model_team = get_fitted_team_model(
        season=season,
        gameweek=min(gameweeks),
        dbsession=dbsession,
        model=team_model,
        **team_model_args,
    )
    logger.info("Calculating fixture score probabilities...")
    fixtures = get_fixtures_for_gameweeks(gameweeks, season=season, dbsession=dbsession)
    fixture_goal_probs = get_goal_probabilities_for_fixtures(
        fixtures, model_team, max_goals=MAX_GOALS
    )

    df_player = get_all_fitted_player_data(
        season, gameweeks[0], model=player_model, dbsession=dbsession
    )

    if include_bonus:
        df_bonus = fit_bonus_points(gameweek=gameweeks[0], season=season)
    else:
        df_bonus = None
    if include_saves:
        df_saves = fit_save_points(gameweek=gameweeks[0], season=season)
    else:
        df_saves = None
    if include_cards:
        df_cards = fit_card_points(gameweek=gameweeks[0], season=season)
    else:
        df_cards = None
    if include_def_con:
        df_def_con = fit_def_con(gameweek=gameweeks[0], season=season)
    else:
        df_def_con = None

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
    gameweeks: list[int] | None = None,
    season: str = CURRENT_SEASON,
    include_bonus: bool = True,
    include_cards: bool = True,
    include_saves: bool = True,
    include_def_con: bool = True,
    tag_prefix: str | None = None,
    player_model: PlayerModel | None = None,
    team_model: TeamModel | None = None,
    team_model_args: dict | None = None,
    dbsession: Session | None = None,
) -> str:
    dbsession = dbsession if dbsession is not None else get_session()
    if team_model_args is None:
        team_model_args = {"epsilon": DEFAULT_TEAM_EPSILON}
    tag = tag_prefix or ""
    tag += str(uuid4())
    if not gameweeks:
        gameweeks = list(range(next_gameweek(), next_gameweek() + 3))
    with console.status("Predicting points..."):
        calc_all_predicted_points(
            gameweeks=gameweeks,
            season=season,
            dbsession=dbsession,
            include_bonus=include_bonus,
            include_cards=include_cards,
            include_saves=include_saves,
            include_def_con=include_def_con,
            tag=tag,
            player_model=player_model,
            team_model=team_model,
            team_model_args=team_model_args,
        )
    return tag


def run_prediction(
    n_gameweeks: int | None,
    gameweek_start: int | None,
    gameweek_end: int | None,
    season: str,
    no_bonus: bool,
    no_cards: bool,
    no_saves: bool,
    team_model_name: str,
    epsilon: float,
    player_model_name: str = "conjugate",
    player_model_options: dict[str, str] | None = None,
    team_model_options: dict[str, str] | None = None,
) -> None:
    """Fill the player prediction database table."""
    gameweeks = get_gameweeks_array(
        n_gameweeks=n_gameweeks,
        gameweek_start=gameweek_start,
        gameweek_end=gameweek_end,
        season=season,
    )
    include_bonus = not no_bonus
    include_cards = not no_cards
    include_saves = not no_saves
    player_model = PLAYER_MODELS.create_with(
        player_model_name, player_model_options or {}
    )
    # --epsilon stays a first-class option because it is the knob people actually
    # tune; anything else goes through --set-team.
    team_options = {"epsilon": str(epsilon), **(team_model_options or {})}
    team_model, team_config = TEAM_MODELS.build(team_model_name, team_options)

    with session_scope() as session:
        session.expire_on_commit = False

        tag = make_predictedscore_table(
            gameweeks=gameweeks,
            season=season,
            include_bonus=include_bonus,
            include_cards=include_cards,
            include_saves=include_saves,
            player_model=player_model,
            team_model=team_model,
            team_model_args=team_config.fit_args(),
            dbsession=session,
        )

        # print players with top predicted points
        get_top_predicted_points(
            gameweeks=gameweeks,
            tag=tag,
            season=season,
            per_position=True,
            n_players=5,
            dbsession=session,
        )
