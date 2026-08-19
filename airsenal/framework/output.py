"""Shared Rich output primitives for AIrsenal commands and workflows."""

from collections.abc import Generator, Iterable, Iterator
from contextlib import contextmanager
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

console = Console()


def print(*objects: Any, **kwargs: Any) -> None:
    """Print objects through the shared Rich console."""
    console.print(*objects, **kwargs)


def table(*columns: str, title: str | None = None) -> Table:
    """Create an AIrsenal-styled Rich table."""
    output_table = Table(title=title, header_style="bold")
    for column in columns:
        output_table.add_column(column)
    return output_table


def _new_progress(*, transient: bool = False) -> Progress:
    """Build a Progress instance with AIrsenal's standard styling."""
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        BarColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        transient=transient,
    )


def track(
    sequence: Iterable[Any],
    *,
    description: str = "Working...",
    total: float | None = None,
    desc: str | None = None,
) -> Iterator[Any]:
    """Iterate over a sequence with a Rich progress bar."""
    if desc is not None:
        description = desc
    with _new_progress() as progress:
        yield from progress.track(sequence, total=total, description=description)


@contextmanager
def progress_bar(*, transient: bool = False) -> Generator[Progress]:
    """Yield an AIrsenal-styled Rich Progress for manual multi-task tracking.

    Use this instead of instantiating `rich.progress.Progress` directly when
    the work being tracked isn't a simple iteration over a sequence, e.g.
    several concurrently-running tasks that each need their own bar.
    """
    with _new_progress(transient=transient) as progress:
        yield progress
