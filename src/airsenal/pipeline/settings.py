"""
Everything about a pipeline run that is not a swappable component.

Nested only where a group earns it: either it has a second consumer (`ChipWeeks`,
which the transfer CLI builds too) or it describes a stage that only sometimes
runs (`DatabaseSettings`). Optimizer settings are not here - they belong to the
optimizer objects.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from airsenal.game.season import (
    CURRENT_SEASON,
    get_past_seasons,
)
from airsenal.optimization.moves import ChipWeeks

# How many gameweeks ahead to look when nothing says otherwise. A default for a
# run, so it lives here with the rest of them rather than in the query layer.
DEFAULT_N_GAMEWEEKS = 3


class StaleDatabase(Enum):
    """What to do when the database could not be brought up to date."""

    # ask, which is what a person at a terminal wants
    ASK = "ask"
    # carry on with what the database already has
    CONTINUE = "continue"
    # stop, which is what an unattended run wants
    ABORT = "abort"


@dataclass(frozen=True)
class DatabaseSettings:
    """Which seasons the database is built from, and whether to rebuild it."""

    clean: bool = False
    n_previous: int = 3
    include_current_season: bool = True

    def seasons(self) -> list[str]:
        """The seasons to fill a fresh database with."""
        past = get_past_seasons(self.n_previous)
        return [CURRENT_SEASON, *past] if self.include_current_season else past


@dataclass(frozen=True)
class PipelineSettings:
    """What a run does, as opposed to what it does it with."""

    fpl_team_id: int | None = None
    n_gameweeks: int = DEFAULT_N_GAMEWEEKS
    season: str = CURRENT_SEASON
    # Where the window starts. None means "the next gameweek", which only the
    # current season can answer - a past season has no next gameweek, so
    # replaying or back-testing one has to say.
    gameweek_start: int | None = None
    # Where it ends, for a caller that names both ends rather than a length.
    # `get_gameweeks_array` rejects a length alongside an end, so at most one of
    # this and n_gameweeks is ever meaningful.
    gameweek_end: int | None = None
    chips: ChipWeeks = field(default_factory=ChipWeeks)
    database: DatabaseSettings = field(default_factory=DatabaseSettings)
    # Whether to create and update the database before predicting. Off, a run
    # works from what the database already holds and makes no network call -
    # which is both a thing people want ("re-optimise without re-fetching") and
    # what makes `run()` runnable from a test.
    refresh_database: bool = True
    on_stale_database: StaleDatabase = StaleDatabase.ASK
    # Whether to build a squad from scratch rather than optimise transfers.
    # None asks the API whether this entry has started yet, which is what the
    # pipeline has always done; replay sets it explicitly instead.
    new_squad: bool | None = None
    # How many free transfers to start from, when the live entry should not be
    # asked. Only the transfer search reads it.
    num_free_transfers: int | None = None
    # Whether a from-scratch build may pick players predicted to score nothing.
    # A property of the candidate pool rather than of any one optimizer, which
    # is why it travels on SquadRequest and therefore has to come from here.
    remove_zero_points_players: bool = True
    # Where to dump every plan the transfer search considered, for debugging.
    save_plans: Path | None = None
    apply_transfers: bool = False
    # Whether to apply without the interactive prompt, for an unattended run.
    skip_confirmation: bool = False
    save_absences: bool = False
