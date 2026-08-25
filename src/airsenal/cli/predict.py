"""Commands for predicting player scores."""

from airsenal.cli import options
from airsenal.db.session import session_scope
from airsenal.pipeline import AIrsenalPipeline, PipelineSettings
from airsenal.prediction.player_models import build_player_model
from airsenal.prediction.points import PointsConfig
from airsenal.prediction.team_models import build_team_model
from airsenal.reporting.top_players import get_top_predicted_points


def predict(
    n_gameweeks: options.OptionalWeeksAhead = None,
    gameweek_start: options.GameweekStart = None,
    gameweek_end: options.GameweekEnd = None,
    season: options.Season = options.DEFAULT_SEASON,
    bonus: options.Bonus = True,
    cards: options.Cards = True,
    saves: options.Saves = True,
    def_con: options.DefCon = True,
    player_model: options.PlayerModel = options.DEFAULT_PLAYER_MODEL,
    team_model: options.TeamModel = options.DEFAULT_TEAM_MODEL,
    epsilon: options.Epsilon = None,
) -> None:
    """Predict player scores for a gameweek range."""
    pipeline = AIrsenalPipeline(
        team_model=build_team_model(team_model, epsilon),
        player_model=build_player_model(player_model),
        points=PointsConfig(bonus=bonus, cards=cards, saves=saves, def_con=def_con),
        settings=PipelineSettings(
            season=season,
            n_gameweeks=n_gameweeks or options.DEFAULT_N_GAMEWEEKS,
            gameweek_start=gameweek_start,
            gameweek_end=gameweek_end,
            refresh_database=False,
        ),
    )
    with session_scope() as session:
        session.expire_on_commit = False
        # the pipeline resolves the window, so a length and a pair of ends are
        # reconciled in one place rather than once per command
        gameweeks = pipeline.gameweeks(session)
        tag = pipeline.predict(gameweeks, session)
        # showing the answer is the command's job, not the prediction's: this was
        # the only prediction -> reporting import in the package
        get_top_predicted_points(
            gameweeks=gameweeks,
            tag=tag,
            season=season,
            per_position=True,
            n_players=5,
            dbsession=session,
        )
