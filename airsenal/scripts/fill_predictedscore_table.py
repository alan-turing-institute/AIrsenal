"""
Fill the "player_prediction" table with score predictions
Usage:
python fill_predictedscore_table.py --weeks_ahead <nweeks>
Generates a "tag" string which is stored so it can later be used by team-optimizers to
get consistent sets of predictions from the database.
"""

from uuid import uuid4

from bpl import ExtendedDixonColesMatchPredictor, NeutralDixonColesMatchPredictor
from sqlalchemy.orm.session import Session

from airsenal.framework.bpl_interface import (
    DEFAULT_TEAM_EPSILON,
    get_fitted_team_model,
    get_goal_probabilities_for_fixtures,
)
from airsenal.framework.output import console, get_logger, track
from airsenal.framework.player_model import ConjugatePlayerModel, NumpyroPlayerModel
from airsenal.framework.prediction_utils import (
    MAX_GOALS,
    calc_predicted_points_for_player,
    fit_bonus_points,
    fit_card_points,
    fit_def_con,
    fit_save_points,
    get_all_fitted_player_data,
)
from airsenal.framework.random_team_model import RandomMatchPredictor
from airsenal.framework.schema import get_session, session_scope
from airsenal.framework.utils import (
    CURRENT_SEASON,
    NEXT_GAMEWEEK,
    get_fixtures_for_gameweek,
    get_gameweeks_array,
    get_top_predicted_points,
    list_players,
)

logger = get_logger(__name__)


def calc_all_predicted_points(
    gw_range: list[int],
    season: str,
    dbsession: Session,
    include_bonus: bool = True,
    include_cards: bool = True,
    include_saves: bool = True,
    include_def_con: bool = True,
    tag: str = "",
    player_model: NumpyroPlayerModel | ConjugatePlayerModel | None = None,
    team_model: ExtendedDixonColesMatchPredictor
    | NeutralDixonColesMatchPredictor
    | RandomMatchPredictor
    | None = None,
    team_model_args: dict | None = None,
) -> None:
    """
    Do the full prediction for players.
    """
    if team_model_args is None:
        team_model_args = {"epsilon": DEFAULT_TEAM_EPSILON}
    model_team = get_fitted_team_model(
        season=season,
        gameweek=min(gw_range),
        dbsession=dbsession,
        model=team_model,
        **team_model_args,
    )
    logger.info("Calculating fixture score probabilities...")
    fixtures = get_fixtures_for_gameweek(gw_range, season=season, dbsession=dbsession)
    fixture_goal_probs = get_goal_probabilities_for_fixtures(
        fixtures, model_team, max_goals=MAX_GOALS
    )

    df_player = get_all_fitted_player_data(
        season, gw_range[0], model=player_model, dbsession=dbsession
    )

    if include_bonus:
        df_bonus = fit_bonus_points(gameweek=gw_range[0], season=season)
    else:
        df_bonus = None
    if include_saves:
        df_saves = fit_save_points(gameweek=gw_range[0], season=season)
    else:
        df_saves = None
    if include_cards:
        df_cards = fit_card_points(gameweek=gw_range[0], season=season)
    else:
        df_cards = None
    if include_def_con:
        df_def_con = fit_def_con(gameweek=gw_range[0], season=season)
    else:
        df_def_con = None

    players = list_players(season=season, gameweek=gw_range[0], dbsession=dbsession)

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
            gw_range=gw_range,
            tag=tag,
            dbsession=dbsession,
        )
        for pred in predictions:
            dbsession.add(pred)
    dbsession.commit()
    logger.info("Finished adding predictions to db")


def make_predictedscore_table(
    gw_range: list[int] | None = None,
    season: str = CURRENT_SEASON,
    include_bonus: bool = True,
    include_cards: bool = True,
    include_saves: bool = True,
    include_def_con: bool = True,
    tag_prefix: str | None = None,
    player_model: NumpyroPlayerModel | ConjugatePlayerModel | None = None,
    team_model: ExtendedDixonColesMatchPredictor
    | NeutralDixonColesMatchPredictor
    | RandomMatchPredictor
    | None = None,
    team_model_args: dict | None = None,
    dbsession: Session | None = None,
) -> str:
    dbsession = dbsession if dbsession is not None else get_session()
    if team_model_args is None:
        team_model_args = {"epsilon": DEFAULT_TEAM_EPSILON}
    tag = tag_prefix or ""
    tag += str(uuid4())
    if not gw_range:
        gw_range = list(range(NEXT_GAMEWEEK, NEXT_GAMEWEEK + 3))
    with console.status("Predicting points..."):
        calc_all_predicted_points(
            gw_range=gw_range,
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
    weeks_ahead: int | None,
    gameweek_start: int | None,
    gameweek_end: int | None,
    season: str,
    no_bonus: bool,
    no_cards: bool,
    no_saves: bool,
    sampling: bool,
    team_model_name: str,
    epsilon: float,
) -> None:
    """Fill the player prediction database table."""
    gw_range = get_gameweeks_array(
        weeks_ahead=weeks_ahead,
        gameweek_start=gameweek_start,
        gameweek_end=gameweek_end,
        season=season,
    )
    include_bonus = not no_bonus
    include_cards = not no_cards
    include_saves = not no_saves
    player_model = NumpyroPlayerModel() if sampling else ConjugatePlayerModel()
    if team_model_name == "extended":
        team_model = ExtendedDixonColesMatchPredictor()
    elif team_model_name == "neutral":
        team_model = NeutralDixonColesMatchPredictor()
    elif team_model_name == "random":
        team_model = RandomMatchPredictor()
    else:
        msg = f"Unknown team model: {team_model_name}"
        raise ValueError(msg)

    with session_scope() as session:
        session.expire_on_commit = False

        tag = make_predictedscore_table(
            gw_range=gw_range,
            season=season,
            include_bonus=include_bonus,
            include_cards=include_cards,
            include_saves=include_saves,
            player_model=player_model,
            team_model=team_model,
            team_model_args={"epsilon": epsilon},
            dbsession=session,
        )

        # print players with top predicted points
        get_top_predicted_points(
            gameweek=gw_range,
            tag=tag,
            season=season,
            per_position=True,
            n_players=5,
            dbsession=session,
        )
