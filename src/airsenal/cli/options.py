"""
The options more than one command takes.

A command signature should reach for a name from this module before writing a
fresh `typer.Option`. Command-specific options stay in their own command: this
module is for the ones that mean the same thing everywhere.
"""

from pathlib import Path
from typing import Annotated

import typer

from airsenal.game.season import CURRENT_SEASON
from airsenal.optimization.squad_optimizers import (
    SQUAD_OPTIMIZERS,
)
from airsenal.optimization.transfer_optimizers import (
    TRANSFER_OPTIMIZERS,
)
from airsenal.pipeline.settings import DEFAULT_N_GAMEWEEKS as _DEFAULT_N_GAMEWEEKS
from airsenal.pipeline.settings import StaleDatabase
from airsenal.prediction.player_models import (
    DEFAULT_PLAYER_MODEL as _DEFAULT_PLAYER_MODEL,
)
from airsenal.prediction.player_models import PLAYER_MODELS
from airsenal.prediction.team_models import DEFAULT_TEAM_MODEL as _DEFAULT_TEAM_MODEL
from airsenal.prediction.team_models import TEAM_MODELS

# Rich help panels, so a command with seventeen options is grouped rather than
# listed in historical-accretion order.
DATABASE = "Database"
PREDICTION = "Prediction"
OPTIMISATION = "Optimisation"
OUTPUT = "Output"

# Defaults re-exported so that no command signature has to restate a value it
# does not own. Each is defined once, beside the setting it is the default for.
DEFAULT_N_GAMEWEEKS = _DEFAULT_N_GAMEWEEKS
DEFAULT_SEASON = CURRENT_SEASON
DEFAULT_N_PREVIOUS = 3
DEFAULT_MAX_TRANSFERS = 2
DEFAULT_MAX_HIT = 8
DEFAULT_NUM_ITERATIONS = 100
# re-exported so a command names one module for both the alias and its default
DEFAULT_PLAYER_MODEL = _DEFAULT_PLAYER_MODEL
DEFAULT_TEAM_MODEL = _DEFAULT_TEAM_MODEL


def _names(table: dict[str, object]) -> str:
    return ", ".join(sorted(table))


# --------------------------------------------------------------- identity ----

FplTeamId = Annotated[
    int | None,
    typer.Option(help="FPL team ID. Defaults to $FPL_TEAM_ID."),
]

Season = Annotated[str, typer.Option(help="Season in the form 2526.")]

# `optimize squad` resolves None differently from the current season: it can
# be asked to build a squad for a past season it is back-testing.
OptionalSeason = Annotated[str | None, typer.Option(help="Season in the form 2526.")]

Tag = Annotated[
    str | None,
    typer.Option(help="Prediction tag; defaults to the latest in the database."),
]

# --------------------------------------------------------- gameweek window ---

# The flag name is pinned rather than derived from the parameter: it is public,
# and the parameters were renamed for consistency with everything else.
WeeksAhead = Annotated[
    int,
    typer.Option("--weeks-ahead", min=1, help="Number of gameweeks to look ahead."),
]

OptionalWeeksAhead = Annotated[
    int | None,
    typer.Option("--weeks-ahead", min=1, help="Number of gameweeks to look ahead."),
]

# `optimize squad` has always spelled this --num-gameweeks; that keeps working,
# but --weeks-ahead is what every other command uses.
SquadWeeksAhead = Annotated[
    int,
    typer.Option(
        "--weeks-ahead",
        "--num-gameweeks",
        min=1,
        help="Number of gameweeks to look ahead.",
    ),
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
        help="Worker processes for the transfer search. Defaults to every core.",
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
    int,
    typer.Option(
        min=1,
        help="How hard to search when rebuilding a squad.",
        rich_help_panel=OPTIMISATION,
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
