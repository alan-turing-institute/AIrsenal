"""Commands that apply AIrsenal recommendations to the FPL API."""

from typing import Annotated

import typer

from airsenal.scripts.make_transfers import make_transfers
from airsenal.scripts.set_lineup import set_lineup

app = typer.Typer(no_args_is_help=True)


@app.command()
def transfers(
    fpl_team_id: Annotated[int | None, typer.Option(help="FPL team ID.")] = None,
    confirm: Annotated[
        bool, typer.Option(help="Skip interactive confirmation.")
    ] = False,
) -> None:
    """Apply suggested transfers and then set the resulting lineup."""
    try:
        make_transfers(fpl_team_id, skip_check=confirm)
        set_lineup(fpl_team_id, skip_check=confirm)
    except Exception as error:
        msg = (
            "Something went wrong when making transfers. Check your team and make "
            "transfers and lineup changes manually on the web-site. If the problem "
            "persists, let us know on GitHub."
        )
        raise RuntimeError(msg) from error


@app.command()
def lineup(
    fpl_team_id: Annotated[int | None, typer.Option(help="FPL team ID.")] = None,
    confirm: Annotated[
        bool, typer.Option(help="Skip interactive confirmation.")
    ] = False,
) -> None:
    """Apply the suggested starting lineup and captain through the FPL API."""
    try:
        set_lineup(fpl_team_id, skip_check=confirm)
    except Exception as error:
        msg = (
            "Something went wrong when setting lineup. Check your lineup manually "
            "on the web-site. If the problem persists, let us know on GitHub."
        )
        raise RuntimeError(msg) from error
