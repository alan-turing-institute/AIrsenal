"""Commands for visualizing FPL league data."""

from importlib import import_module
from typing import Annotated

import typer


def plot(
    metric: Annotated[
        str,
        typer.Option(help="points, total_points, ranking, or overall_ranking."),
    ] = "total_points",
) -> None:
    """Plot a mini-league metric by gameweek."""
    plot_standings = import_module("airsenal.reporting.plots").plot_standings
    plot_standings(metric)
