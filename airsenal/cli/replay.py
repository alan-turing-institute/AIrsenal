"""Commands for replaying historical seasons."""

import typer

from airsenal.framework.bpl_interface import DEFAULT_TEAM_EPSILON
from airsenal.scripts.replay_season import run_replays


def replay(
    season: str = typer.Option(..., help="Season in the form 2526."),
    gameweek_start: int = typer.Option(1, min=1, help="First gameweek to replay."),
    gameweek_end: int | None = typer.Option(None, help="Last gameweek to replay."),
    weeks_ahead: int = typer.Option(3, min=1, help="Prediction horizon per gameweek."),
    fpl_team_id: int | None = typer.Option(None, help="FPL team ID for the replay."),
    resume: bool = typer.Option(False, help="Resume an existing replay team."),
    num_thread: int = typer.Option(4, min=1, help="Worker processes to use."),
    loop: int = typer.Option(1, help="Replay count; -1 repeats indefinitely."),
    team_model: str = typer.Option("extended", help="Team model: extended or random."),
    epsilon: float = typer.Option(DEFAULT_TEAM_EPSILON, help="Time-weighting factor."),
    max_transfers: int = typer.Option(2, min=0, help="Maximum transfers per gameweek."),
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
