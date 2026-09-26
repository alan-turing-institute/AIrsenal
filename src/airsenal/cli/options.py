"""
Command-line options that mean the same thing wherever they appear.

A command signature should reach for a name from this module before writing a
fresh `typer.Option`. An option only one command could take stays in that
command.

Types only: an option's default belongs to whatever the option configures, so a
command imports it from there rather than from here.
"""

from typing import Annotated, Any

import typer

from airsenal.optimization.squad_optimizers import (
    SQUAD_OPTIMIZERS,
)
from airsenal.optimization.transfer_optimizers import (
    TRANSFER_OPTIMIZERS,
)
from airsenal.prediction.minutes_models import MINUTES_MODELS
from airsenal.prediction.player_models import PLAYER_MODELS
from airsenal.prediction.points_models import POINTS_MODELS
from airsenal.prediction.team_models import TEAM_MODELS

# Rich help panels, so commands can be grouped rather than listed as one flat block.
DATABASE = "Database"
PREDICTION = "Prediction"
OPTIMISATION = "Optimisation"
OUTPUT = "Output"


_SEASON_HELP = "Season in the form 2526."
_N_GAMEWEEKS_HELP = "Number of gameweeks to look ahead."


def _names(table: dict[str, object]) -> str:
    return ", ".join(sorted(table))


def _chip_option(chip: str) -> Any:
    """The option naming the gameweek to play `chip` in."""
    return typer.Option(help=f"{chip} gameweek; 0 for any gameweek, -1 for never.")


# --------------------------------------------------------------- identity ----

FplTeamId = Annotated[
    int | None,
    typer.Option(help="FPL team ID. Defaults to $FPL_TEAM_ID."),
]

Season = Annotated[str, typer.Option(help=_SEASON_HELP)]
OptionalSeason = Annotated[str | None, typer.Option(help=_SEASON_HELP)]

Tag = Annotated[
    str | None,
    typer.Option(help="Prediction tag; defaults to the latest in the database."),
]

# --------------------------------------------------------- gameweek window ---

NGameweeks = Annotated[
    int,
    typer.Option("--n-gameweeks", min=1, help=_N_GAMEWEEKS_HELP),
]

OptionalNGameweeks = Annotated[
    int | None,
    typer.Option("--n-gameweeks", min=1, help=_N_GAMEWEEKS_HELP),
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

MinutesModel = Annotated[
    str,
    typer.Option(
        help=f"Minutes model: {_names(MINUTES_MODELS)}.", rich_help_panel=PREDICTION
    ),
]

PointsModel = Annotated[
    str,
    typer.Option(
        help=(
            f"Points model: {_names(POINTS_MODELS)}. The team, player and "
            "minutes models are parts of the component one."
        ),
        rich_help_panel=PREDICTION,
    ),
]

Epsilon = Annotated[
    float | None,
    typer.Option(
        help=(
            "Exponential time-weighting downweight factor for the team model. "
            "Defaults to that model's own value; the player model's weighting "
            "is separate and has no flag."
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

MaxExpansions = Annotated[
    int | None,
    typer.Option(
        min=1,
        help=(
            "Most nodes the MCTS makes. Defaults to its own value; the tree search "
            "makes every node and rejects it."
        ),
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

# ------------------------------------------------------------------ chips ----

WildcardGameweek = Annotated[int, _chip_option("Wildcard")]
FreeHitGameweek = Annotated[int, _chip_option("Free hit")]
TripleCaptainGameweek = Annotated[int, _chip_option("Triple captain")]
BenchBoostGameweek = Annotated[int, _chip_option("Bench boost")]
ChipHeuristic = Annotated[
    bool,
    typer.Option(
        help=(
            "Choose each chip's gameweek by rules about the fixtures, in place of "
            "the four chip gameweek options."
        )
    ),
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
