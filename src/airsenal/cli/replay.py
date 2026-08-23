"""Commands for replaying historical seasons."""

from typing import Annotated

import typer

from airsenal.cli.options import parse_options
from airsenal.optimization.moves import TransferConstraints
from airsenal.optimization.transfer_optimizers import (
    TRANSFER_OPTIMIZERS,
    TreeSearchConfig,
)
from airsenal.pipeline import (
    AIrsenalPipeline,
    PipelineSettings,
    ReplaySettings,
    run_replays,
)
from airsenal.prediction.registry import (
    DEFAULT_PLAYER_MODEL,
    DEFAULT_TEAM_MODEL,
    PLAYER_MODELS,
    TEAM_MODELS,
)


def replay(
    season: Annotated[str, typer.Option(help="Season in the form 2526.")],
    gameweek_start: Annotated[
        int, typer.Option(min=1, help="First gameweek to replay.")
    ] = 1,
    gameweek_end: Annotated[
        int | None, typer.Option(help="Last gameweek to replay.")
    ] = None,
    n_gameweeks: Annotated[
        int,
        typer.Option("--weeks-ahead", min=1, help="Prediction horizon per gameweek."),
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
        str, typer.Option(help=f"Team model: {', '.join(TEAM_MODELS.names())}.")
    ] = DEFAULT_TEAM_MODEL,
    epsilon: Annotated[
        float | None,
        typer.Option(
            help="Time-weighting factor. Defaults to the team model's own value."
        ),
    ] = None,
    max_transfers: Annotated[
        int, typer.Option(min=0, help="Maximum transfers per gameweek.")
    ] = 2,
    player_model: Annotated[
        str,
        typer.Option(help=f"Player model: {', '.join(PLAYER_MODELS.names())}."),
    ] = DEFAULT_PLAYER_MODEL,
    set_player: Annotated[
        list[str] | None,
        typer.Option("--set-player", help="Player model option as key=value."),
    ] = None,
    set_team: Annotated[
        list[str] | None,
        typer.Option("--set-team", help="Team model option as key=value."),
    ] = None,
) -> None:
    """Replay a historical FPL season."""
    run_replays(
        AIrsenalPipeline.from_names(
            team_model=team_model,
            player_model=player_model,
            epsilon=epsilon,
            team_options=parse_options(set_team),
            player_options=parse_options(set_player),
            transfer_optimizer=TRANSFER_OPTIMIZERS.create(
                "tree_search", TreeSearchConfig(num_thread=num_thread)
            ),
            constraints=TransferConstraints(max_opt_transfers=max_transfers),
            settings=PipelineSettings(
                fpl_team_id=fpl_team_id,
                n_gameweeks=n_gameweeks,
                season=season,
                # replay never touches the real entry or the live API
                refresh_database=False,
                apply_transfers=False,
            ),
        ),
        ReplaySettings(
            gameweek_start=gameweek_start,
            gameweek_end=gameweek_end,
            loop=loop,
            resume=resume,
        ),
    )
