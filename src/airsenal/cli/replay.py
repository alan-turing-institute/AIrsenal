"""Commands for replaying historical seasons."""

from typing import Annotated

import typer

from airsenal.prediction.team_models.dixon_coles import DEFAULT_TEAM_EPSILON
from airsenal.scripts.replay_season import run_replays


def replay(
    season: Annotated[str, typer.Option(help="Season in the form 2526.")],
    gameweek_start: Annotated[
        int, typer.Option(min=1, help="First gameweek to replay.")
    ] = 1,
    gameweek_end: Annotated[
        int | None, typer.Option(help="Last gameweek to replay.")
    ] = None,
    weeks_ahead: Annotated[
        int, typer.Option(min=1, help="Prediction horizon per gameweek.")
    ] = 3,
    fpl_team_id: Annotated[
        int | None, typer.Option(help="FPL team ID for the replay.")
    ] = None,
    resume: Annotated[
        bool, typer.Option(help="Resume an existing replay team.")
    ] = False,
    num_thread: Annotated[
        int, typer.Option(min=1, help="Worker processes to use.")
    ] = 4,
    loop: Annotated[
        int, typer.Option(help="Replay count; -1 repeats indefinitely.")
    ] = 1,
    team_model: Annotated[
        str, typer.Option(help="Team model: extended or random.")
    ] = "extended",
    epsilon: Annotated[
        float, typer.Option(help="Time-weighting factor.")
    ] = DEFAULT_TEAM_EPSILON,
    max_transfers: Annotated[
        int, typer.Option(min=0, help="Maximum transfers per gameweek.")
    ] = 2,
) -> None:
    """Replay a historical FPL season."""
    run_replays(
        season,
        gameweek_start,
        gameweek_end,
        weeks_ahead,
        fpl_team_id,
        resume,
        num_thread,
        loop,
        team_model,
        epsilon,
        max_transfers,
    )
