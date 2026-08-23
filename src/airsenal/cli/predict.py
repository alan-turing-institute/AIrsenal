"""Commands for predicting player scores."""

from typing import Annotated

import typer

from airsenal.core.season import CURRENT_SEASON
from airsenal.prediction.player_models import (
    DEFAULT_PLAYER_MODEL,
    PLAYER_MODELS,
)
from airsenal.prediction.run import run_prediction
from airsenal.prediction.team_models import (
    DEFAULT_TEAM_MODEL,
    TEAM_MODELS,
)


def predict(
    n_gameweeks: Annotated[
        int | None,
        typer.Option("--weeks-ahead", help="Number of gameweeks to predict."),
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
    player_model: Annotated[
        str,
        typer.Option(help=f"Player model: {', '.join(sorted(PLAYER_MODELS))}."),
    ] = DEFAULT_PLAYER_MODEL,
    team_model: Annotated[
        str,
        typer.Option(help=f"Team model: {', '.join(sorted(TEAM_MODELS))}."),
    ] = DEFAULT_TEAM_MODEL,
    epsilon: Annotated[
        float | None,
        typer.Option(
            help=(
                "Exponential time-weighting downweight factor. "
                "Defaults to the team model's own value."
            )
        ),
    ] = None,
) -> None:
    """Predict player scores for a gameweek range."""
    run_prediction(
        n_gameweeks=n_gameweeks,
        gameweek_start=gameweek_start,
        gameweek_end=gameweek_end,
        season=season,
        no_bonus=no_bonus,
        no_cards=no_cards,
        no_saves=no_saves,
        player_model_name=player_model,
        team_model_name=team_model,
        epsilon=epsilon,
    )
