#!/usr/bin/env python3
import click

from airsenal.scripts.make_transfers import make_transfers
from airsenal.scripts.set_lineup import set_lineup


@click.command()
@click.option(
    "--transfers",
    is_flag=True,
    help="Make the transfers on FPL website. NOTE: This is not reversible.",
)
@click.option("--lineup", is_flag=True, help="Set suggested lineup on the FPL website.")
@click.option("--fpl-team-id", type=int, help="FPL team ID.")
@click.option(
    "--noconfirm",
    is_flag=True,
    help="If set, doesn't ask for confirmation to make changes to FPL website.",
)
def apply(transfers, lineup, fpl_team_id, noconfirm):
    """
    Update transfers and lineup on FPL website.
    """
    if transfers:
        make_transfers(fpl_team_id, noconfirm)
    if lineup:
        set_lineup(fpl_team_id, noconfirm)
