"""
The options more than one command takes.

A command signature should reach for a name from this module before writing a
fresh `typer.Option`. Command-specific options stay in their own command: this
module is for the ones that mean the same thing everywhere.

Types only: an option's default belongs to whatever the option configures, so a
command imports it from there rather than from here.
"""

from pathlib import Path
from typing import Annotated

import typer

from airsenal.optimization.squad_optimizers import (
    SQUAD_OPTIMIZERS,
)
from airsenal.optimization.transfer_optimizers import (
    TRANSFER_OPTIMIZERS,
)
from airsenal.pipeline.settings import StaleDatabase
from airsenal.prediction.player_models import PLAYER_MODELS
from airsenal.prediction.team_models import TEAM_MODELS

# Rich help panels, so commands can be grouped rather than listed as one flat block.
DATABASE = "Database"
PREDICTION = "Prediction"
OPTIMISATION = "Optimisation"
OUTPUT = "Output"


def _names(table: dict[str, object]) -> str:
    return ", ".join(sorted(table))


# --------------------------------------------------------------- identity ----

FplTeamId = Annotated[
    int | None,
    typer.Option(help="FPL team ID. Defaults to $FPL_TEAM_ID."),
]

Season = Annotated[str, typer.Option(help="Season in the form 2526.")]
OptionalSeason = Annotated[str | None, typer.Option(help="Season in the form 2526.")]

Tag = Annotated[
    str | None,
    typer.Option(help="Prediction tag; defaults to the latest in the database."),
]

# --------------------------------------------------------- gameweek window ---

NGameweeks = Annotated[
    int,
    typer.Option("--n-gameweeks", min=1, help="Number of gameweeks to look ahead."),
]

OptionalNGameweeks = Annotated[
    int | None,
    typer.Option("--n-gameweeks", min=1, help="Number of gameweeks to look ahead."),
]

GameweekStart = Annotated[int | None, typer.Option(help="First gameweek to cover.")]

GameweekEnd = Annotated[int | None, typer.Option(help="Last gameweek to cover.")]

# --------------------------------------------------------------- database ----

Clean = Annotated[
    bool,
    typer.Option(help="Delete and recreate the database.", rich_help_panel=DATABASE),
]

NPrevious = Annotated[
    int,
    typer.Option(
        min=0,
        help="Number of previous seasons to include.",
        rich_help_panel=DATABASE,
    ),
]

CurrentSeason = Annotated[
    bool,
    typer.Option(
        help="Include the current season in a fresh database.",
        rich_help_panel=DATABASE,
    ),
]

RefreshDatabase = Annotated[
    bool,
    typer.Option(
        help=(
            "Fetch new data before predicting. Off, the run works from what the "
            "database already holds - re-optimise without re-fetching."
        ),
        rich_help_panel=DATABASE,
    ),
]

OnStale = Annotated[
    StaleDatabase,
    typer.Option(
        help=(
            "What to do if the database could not be brought up to date. "
            "'abort' is the one for an unattended run."
        ),
        rich_help_panel=DATABASE,
    ),
]

# ------------------------------------------------------------- prediction ----

PlayerModel = Annotated[
    str,
    typer.Option(
        help=f"Player model: {_names(PLAYER_MODELS)}.", rich_help_panel=PREDICTION
    ),
]

TeamModel = Annotated[
    str,
    typer.Option(
        help=f"Team model: {_names(TEAM_MODELS)}.", rich_help_panel=PREDICTION
    ),
]

Epsilon = Annotated[
    float | None,
    typer.Option(
        help=(
            "Exponential time-weighting downweight factor. Defaults to the team "
            "model's own value."
        ),
        rich_help_panel=PREDICTION,
    ),
]

Bonus = Annotated[
    bool,
    typer.Option(help="Include bonus points.", rich_help_panel=PREDICTION),
]

Cards = Annotated[
    bool,
    typer.Option(help="Include card-point deductions.", rich_help_panel=PREDICTION),
]

Saves = Annotated[
    bool,
    typer.Option(help="Include goalkeeper save points.", rich_help_panel=PREDICTION),
]

DefCon = Annotated[
    bool,
    typer.Option(
        help="Include defensive-contribution points.", rich_help_panel=PREDICTION
    ),
]

# ----------------------------------------------------------- optimisation ----

TransferOptimizer = Annotated[
    str,
    typer.Option(
        help=f"Transfer search: {_names(TRANSFER_OPTIMIZERS)}.",
        rich_help_panel=OPTIMISATION,
    ),
]

SquadOptimizer = Annotated[
    str,
    typer.Option(
        help=(
            "Whole-squad optimizer, used for a from-scratch squad and for a "
            f"wildcard or free hit: {_names(SQUAD_OPTIMIZERS)}."
        ),
        rich_help_panel=OPTIMISATION,
    ),
]

MaxTransfers = Annotated[
    int,
    typer.Option(
        min=0,
        help="Maximum transfers to consider per gameweek.",
        rich_help_panel=OPTIMISATION,
    ),
]

MaxHit = Annotated[
    int,
    typer.Option(
        min=0,
        help="Maximum points to spend on transfers.",
        rich_help_panel=OPTIMISATION,
    ),
]

AllowUnused = Annotated[
    bool,
    typer.Option(
        help="Consider plans that waste free transfers.",
        rich_help_panel=OPTIMISATION,
    ),
]

Subs = Annotated[
    bool,
    typer.Option(
        help="Count substitutes' predicted points.", rich_help_panel=OPTIMISATION
    ),
]

NumThread = Annotated[
    int | None,
    typer.Option(
        min=1,
        help="Worker processes for the transfer search.",
        rich_help_panel=OPTIMISATION,
    ),
]

NumFreeTransfers = Annotated[
    int | None,
    typer.Option(
        min=0,
        max=5,
        help="Free transfers available. Defaults to asking the FPL API.",
        rich_help_panel=OPTIMISATION,
    ),
]

Budget = Annotated[
    int,
    typer.Option(
        min=0, help="Budget in 0.1 million units.", rich_help_panel=OPTIMISATION
    ),
]

NumGenerations = Annotated[
    int | None,
    typer.Option(
        min=1,
        help="Genetic algorithm generations.",
        rich_help_panel=OPTIMISATION,
    ),
]

PopulationSize = Annotated[
    int | None,
    typer.Option(
        min=1,
        help="Candidate squads per generation.",
        rich_help_panel=OPTIMISATION,
    ),
]

ZeroPointsPlayers = Annotated[
    bool,
    typer.Option(
        help="Consider players predicted to score nothing.",
        rich_help_panel=OPTIMISATION,
    ),
]

NumIterations = Annotated[
    int | None,
    typer.Option(
        min=1,
        help=(
            "How hard to search when rebuilding a squad. Defaults to the "
            "optimizer's own value."
        ),
        rich_help_panel=OPTIMISATION,
    ),
]

DryRun = Annotated[
    bool,
    typer.Option(
        help="Show what would be sent to the FPL API, and send nothing.",
        rich_help_panel=OUTPUT,
    ),
]

OutputDir = Annotated[
    Path | None,
    typer.Option(
        help="Directory to write results to. Defaults to the current directory.",
        rich_help_panel=OUTPUT,
    ),
]

TagPrefix = Annotated[
    str,
    typer.Option(
        help="Prefix for the result tag and filename. Defaults to a timestamped one.",
        rich_help_panel=OUTPUT,
    ),
]


# ------------------------------------------------------------------ chips ----

WildcardWeek = Annotated[
    int,
    typer.Option(help="Wildcard week; 0 for any week, -1 for never."),
]

FreeHitWeek = Annotated[
    int,
    typer.Option(help="Free hit week; 0 for any week, -1 for never."),
]

TripleCaptainWeek = Annotated[
    int,
    typer.Option(help="Triple captain week; 0 for any week, -1 for never."),
]

BenchBoostWeek = Annotated[
    int,
    typer.Option(help="Bench boost week; 0 for any week, -1 for never."),
]

# ----------------------------------------------------------------- output ----

Yes = Annotated[
    bool,
    typer.Option(
        "--yes",
        "-y",
        help="Do not ask for confirmation before applying anything.",
    ),
]

ApplyTransfers = Annotated[
    bool,
    typer.Option(
        help="Apply the suggested transfers and lineup through the FPL API.",
        rich_help_panel=OUTPUT,
    ),
]

Profile = Annotated[
    bool,
    typer.Option(help="Profile the search's execution time.", rich_help_panel=OUTPUT),
]

SaveAbsences = Annotated[
    bool,
    typer.Option(help="Save expected absences to a CSV file.", rich_help_panel=OUTPUT),
]

SavePlans = Annotated[
    Path | None,
    typer.Option(
        help="Directory to write every plan considered to, as JSON.",
        rich_help_panel=OUTPUT,
    ),
]

# An internal persistence mode, not a user concept: `replay` sets it, and a
# person running `optimize` has no reason to.
IsReplay = Annotated[
    bool,
    typer.Option(help="Store suggestions as replay transactions.", hidden=True),
]
