"""Commands for predicting player scores."""

from typing import Annotated

import typer

from airsenal.domain.season import CURRENT_SEASON
from airsenal.prediction.team_models.dixon_coles import DEFAULT_TEAM_EPSILON
from airsenal.scripts.fill_predictedscore_table import run_prediction


def predict(
    weeks_ahead: Annotated[
        int | None, typer.Option(help="Number of gameweeks to predict.")
    ] = None,
    gameweek_start: Annotated[
        int | None, typer.Option(help="First gameweek to predict.")
    ] = None,
    gameweek_end: Annotated[
        int | None, typer.Option(help="Last gameweek to predict.")
    ] = None,
    season: Annotated[
        str, typer.Option(help="Season in the form 2526.")
    ] = CURRENT_SEASON,
    no_bonus: Annotated[
        bool, typer.Option(help="Exclude bonus-point predictions.")
    ] = False,
    no_cards: Annotated[
        bool, typer.Option(help="Exclude card-point deductions.")
    ] = False,
    no_saves: Annotated[
        bool, typer.Option(help="Exclude goalkeeper save points.")
    ] = False,
    sampling: Annotated[
        bool, typer.Option(help="Fit the player model with Numpyro.")
    ] = False,
    team_model: Annotated[
        str, typer.Option(help="Team model: extended, neutral, or random.")
    ] = "extended",
    epsilon: Annotated[
        float, typer.Option(help="Exponential time-weighting downweight factor.")
    ] = DEFAULT_TEAM_EPSILON,
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
