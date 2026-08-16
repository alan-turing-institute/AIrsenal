"""Commands for predicting player scores."""

import typer

from airsenal.framework.bpl_interface import DEFAULT_TEAM_EPSILON
from airsenal.framework.utils import CURRENT_SEASON
from airsenal.scripts.fill_predictedscore_table import run_prediction


def predict(
    weeks_ahead: int | None = typer.Option(
        None, help="Number of gameweeks to predict."
    ),
    gameweek_start: int | None = typer.Option(None, help="First gameweek to predict."),
    gameweek_end: int | None = typer.Option(None, help="Last gameweek to predict."),
    season: str = typer.Option(CURRENT_SEASON, help="Season in the form 2526."),
    no_bonus: bool = typer.Option(False, help="Exclude bonus-point predictions."),
    no_cards: bool = typer.Option(False, help="Exclude card-point deductions."),
    no_saves: bool = typer.Option(False, help="Exclude goalkeeper save points."),
    sampling: bool = typer.Option(False, help="Fit the player model with Numpyro."),
    team_model: str = typer.Option(
        "extended", help="Team model: extended, neutral, or random."
    ),
    epsilon: float = typer.Option(
        DEFAULT_TEAM_EPSILON,
        help="Exponential time-weighting downweight factor.",
    ),
) -> None:
    """Predict player scores for a gameweek range."""
    run_prediction(
        weeks_ahead=weeks_ahead,
        gameweek_start=gameweek_start,
        gameweek_end=gameweek_end,
        season=season,
        no_bonus=no_bonus,
        no_cards=no_cards,
        no_saves=no_saves,
        sampling=sampling,
        team_model_name=team_model,
        epsilon=epsilon,
    )
