"""Commands for predicting player scores."""

from airsenal.cli import options
from airsenal.db.session import session_scope
from airsenal.prediction.run import run_prediction
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
    with session_scope() as session:
        session.expire_on_commit = False
        gameweeks, tag = run_prediction(
            # a window is a length or a pair of ends, never both - the same
            # resolution `optimize transfers` and AIrsenalPipeline.gameweeks do
            n_gameweeks=(
                None
                if gameweek_end is not None
                else (n_gameweeks or options.DEFAULT_N_GAMEWEEKS)
            ),
            gameweek_start=gameweek_start,
            gameweek_end=gameweek_end,
            season=season,
            include_bonus=bonus,
            include_cards=cards,
            include_saves=saves,
            include_def_con=def_con,
            player_model_name=player_model,
            team_model_name=team_model,
            epsilon=epsilon,
            dbsession=session,
        )
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
