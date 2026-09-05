"""Everything about a pipeline run that is not a swappable component."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from airsenal.game.season import (
    CURRENT_SEASON,
    get_past_seasons,
)
from airsenal.optimization.moves import ChipWeeks

# How many gameweeks ahead to look when nothing says otherwise.
DEFAULT_N_GAMEWEEKS = 3

# How many past seasons a fresh database is filled with when nothing says otherwise.
DEFAULT_N_PREVIOUS = 3


class StaleDatabase(Enum):
    """What to do when the database could not be brought up to date."""

    ASK = "ask"
    CONTINUE = "continue"
    ABORT = "abort"


@dataclass(frozen=True)
class DatabaseSettings:
    """Which seasons the database is built from, and whether to rebuild it."""

    clean: bool = False
    n_previous: int = DEFAULT_N_PREVIOUS
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
    # Where the window starts. None means the next gameweek (valid for current season
    # only).
    gameweek_start: int | None = None
    # Where it ends, inclusive, for a caller that names both ends rather than a
    # length. Only one of n_gameweeks and gameweek_end should be set.
    gameweek_end: int | None = None
    chips: ChipWeeks = field(default_factory=ChipWeeks)
    database: DatabaseSettings = field(default_factory=DatabaseSettings)
    # Whether to create and update the database before predicting.
    refresh_database: bool = True
    on_stale_database: StaleDatabase = StaleDatabase.ASK
    # Whether to build a squad from scratch rather than optimise transfers. If None,
    # check whether the API says the entry has started yet.
    new_squad: bool | None = None
    # How many free transfers to start from, or None to use the API's number.
    num_free_transfers: int | None = None
    # Whether a from-scratch build may pick players predicted to score nothing (reduces
    # the search space and is unlikely to impact the final result).
    remove_zero_points_players: bool = True
    # Where to dump every plan the transfer search considered, for debugging.
    save_plans: Path | None = None
    apply_transfers: bool = False
    # Whether to apply without the interactive prompt, for an unattended run.
    skip_confirmation: bool = False
    save_absences: bool = False
