"""Commands that apply AIrsenal recommendations to the FPL API."""

import typer

from airsenal.apply.lineup import set_lineup
from airsenal.apply.transfers import make_transfers
from airsenal.cli import options

app = typer.Typer(
    no_args_is_help=True, help="Apply AIrsenal recommendations to your FPL team."
)


@app.command()
def transfers(
    fpl_team_id: options.FplTeamId = None,
    yes: options.Yes = False,
    dry_run: options.DryRun = False,
) -> None:
    """Apply suggested transfers and then set the resulting lineup."""
    try:
        make_transfers(fpl_team_id, skip_check=yes, dry_run=dry_run)
        set_lineup(fpl_team_id, skip_check=yes, dry_run=dry_run)
    except Exception as error:
        msg = (
            "Something went wrong when making transfers. Check your team and make "
            "transfers and lineup changes manually on the web-site. If the problem "
            "persists, let us know on GitHub."
        )
        raise RuntimeError(msg) from error


@app.command()
def lineup(
    fpl_team_id: options.FplTeamId = None,
    yes: options.Yes = False,
    dry_run: options.DryRun = False,
) -> None:
    """Apply the suggested starting lineup and captain through the FPL API."""
    try:
        set_lineup(fpl_team_id, skip_check=yes, dry_run=dry_run)
    except Exception as error:
        msg = (
            "Something went wrong when setting lineup. Check your lineup manually "
            "on the web-site. If the problem persists, let us know on GitHub."
        )
        raise RuntimeError(msg) from error
