#!/usr/bin/env python3

import click

from airsenal.framework.utils import CURRENT_SEASON
from airsenal.scripts.fill_predictedscore_table import make_predictions


@click.command()
@click.option(
    "--weeks-ahead", type=int, help="Number of weeks to predict into the future."
)
@click.option(
    "--gameweek-start", type=int, help="First gameweek to start looking from."
)
@click.option(
    "--gameweek-end",
    type=int,
    help="Last gameweek to look at. If not given, 3 weeks are used.",
)
@click.option("--season", default=CURRENT_SEASON, help="Season, in format '1819'.")
@click.option(
    "--num-threads",
    default=1,
    help="Number of threads to parallelise over (default 2).",
)
@click.option("--no-bonus", is_flag=True, help="If set, don't include bonus points.")
@click.option(
    "--no-cards",
    is_flag=True,
    help="If set, don't include points lost to yellow and red cards.",
)
@click.option(
    "--no-saves",
    is_flag=True,
    help="If set, don't include save points for goalkeepers.",
)
@click.option(
    "--sampling",
    is_flag=True,
    help="If set, fit the model using sampling with numpyro.",
)
def predict(
    weeks_ahead: int,
    gameweek_start: int,
    gameweek_end: int,
    season: int,
    num_threads: int,
    no_bonus: bool,
    no_cards: bool,
    no_saves: bool,
    sampling: bool,
):
    """
    Fit the data and predict the expected value of points.

    Given gameweeks, this method fits the model to the data and predicts
    the expected value of points for each player.
    """
    make_predictions(
        weeks_ahead,
        gameweek_start,
        gameweek_end,
        season,
        num_threads,
        no_bonus,
        no_cards,
        no_saves,
        sampling,
    )
