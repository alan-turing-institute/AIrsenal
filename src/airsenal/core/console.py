"""Rich console, tables and progress bars: everything that renders."""

import os
import threading
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


def _reset_console_after_fork() -> None:
    """Give a forked child its own console state.

    A Rich `Live` display (a progress bar, or `console.status`) refreshes on a
    background thread that holds `console._lock` while it renders. Fork while it
    holds the lock and the child inherits a locked lock whose owner thread does
    not exist there, so the child blocks forever the first time it logs - which
    every transfer-optimisation worker does. The parent is none the wiser: the
    worker is still alive, so nothing raises, and the run hangs with the
    progress bar frozen part-way.

    `logging` reinitialises its own handler locks at fork for exactly this
    reason; Rich does not, so do it here. The live stack and render hooks are
    dropped too: they describe a display the parent owns, and a child that
    tried to redraw it would garble the shared terminal.
    """
    console._lock = threading.RLock()
    console._record_buffer_lock = threading.RLock()
    del console._buffer[:]
    console._buffer_index = 0
    console._live_stack.clear()
    console._render_hooks.clear()


if hasattr(os, "register_at_fork"):  # pragma: no branch - posix only
    os.register_at_fork(after_in_child=_reset_console_after_fork)


def table(*columns: str, title: str | None = None) -> Table:
    """Create an AIrsenal-styled Rich table."""
    output_table = Table(title=title, header_style="bold")
    for column in columns:
        output_table.add_column(column)
    return output_table


def price_str(price: int | None) -> str:
    """Format a player price (in tenths of a million) as e.g. ``£5.5m``."""
    return f"£{price / 10}m" if price is not None else "-"


def _new_progress(*, transient: bool = False) -> Progress:
    """Build a Progress instance with AIrsenal's standard styling.

    Explicitly bound to our shared `console` rather than Rich's own global
    default - otherwise this Progress's Live display and any other Live
    display elsewhere in AIrsenal (e.g. `console.status(...)`) end up on two
    separate, uncoordinated Live stacks that both try to control the
    terminal at once, which shows up as flickering between the two.
    """
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        BarColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        transient=transient,
        console=console,
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
