"""Commands for visualizing FPL league data."""

from importlib import import_module

import typer


def plot(
    metric: str = typer.Option(
        "total_points", help="points, total_points, ranking, or overall_ranking."
    ),
) -> None:
    """Plot a mini-league metric by gameweek."""
    plot_standings = import_module(
        "airsenal.scripts.plot_league_standings"
    ).plot_standings
    plot_standings(metric)
