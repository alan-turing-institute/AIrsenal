"""Shared Rich output primitives for AIrsenal commands and workflows."""

import io
import logging
import os
import sys
import threading
from collections.abc import Generator, Iterable, Iterator
from contextlib import contextmanager
from typing import Any, TextIO

from rich.console import Console
from rich.logging import RichHandler
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


def _fresh_stream(stream: TextIO) -> TextIO:
    """A new writer onto the same destination, with its own lock and buffer.

    `os.dup` gives a second descriptor for the same open file, so output still
    lands on the same terminal; what is *not* shared is the Python-level
    `BufferedWriter` and the lock inside it.
    """
    return io.TextIOWrapper(
        io.FileIO(os.dup(stream.fileno()), "w", closefd=True),
        encoding=stream.encoding,
        errors=stream.errors,
        line_buffering=True,
    )


def _reset_console_after_fork() -> None:
    """Give a forked child its own console, and its own way of writing.

    Two locks are inherited by a child that forks while the terminal is being
    written to, and either one wedges it permanently, because the thread that
    held the lock does not exist on the other side of the fork:

    1. `console._lock`, held by a Rich `Live` display (a progress bar, or
       `console.status`) for the whole of a refresh.
    2. The lock inside `sys.stdout`'s `BufferedWriter`, held for the actual
       write - which Rich performs while holding the lock above, and which is
       slow, because it is a write to a terminal. CPython has never sanitised
       its io locks across fork (python/cpython#50970), so this one is the
       wider window of the two by far.

    Both are silent: the worker stays alive, so nothing raises, and the run
    stops with the progress bar frozen part-way. `logging` reinitialises its
    own handler locks at fork for exactly this reason; nothing reinitialises
    these, so do it here.

    The live stack and render hooks are dropped too: they describe a display
    the parent owns, and a child that tried to redraw it would garble the
    shared terminal. Discarding the inherited output buffer is deliberate for
    the same reason - it holds bytes the parent has not written yet, and the
    parent will write them itself.
    """
    console._lock = threading.RLock()
    console._record_buffer_lock = threading.RLock()
    del console._buffer[:]
    console._buffer_index = 0
    console._live_stack.clear()
    console._render_hooks.clear()

    # console.file follows sys.stdout unless one was passed explicitly, so
    # replacing the interpreter's streams is enough for Rich as well.
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name)
        try:
            setattr(sys, name, _fresh_stream(stream))
        except (AttributeError, OSError, ValueError):
            # No real file underneath - pytest's capture, say. Nothing to fix:
            # such a stream is not the inherited BufferedWriter either.
            continue


if hasattr(os, "register_at_fork"):  # pragma: no branch - posix only
    os.register_at_fork(after_in_child=_reset_console_after_fork)


_LOGGER_NAME = "airsenal"


def configure_logging(level: int | str = logging.INFO) -> None:
    """Configure the AIrsenal logger to write through Rich.

    Designed for a CLI user reading a terminal, not an operator reading a log
    file: no timestamps, logger names, or file paths - just the message,
    colour-coded by level, with Rich markup in the message rendered.

    Parameters
    ----------
    level : int | str
        Minimum level to display (e.g. ``logging.DEBUG`` or ``"DEBUG"``).
    """
    handler = RichHandler(
        console=console,
        show_time=False,
        show_level=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
    )
    handler.setFormatter(logging.Formatter("%(message)s"))

    logger = logging.getLogger(_LOGGER_NAME)
    logger.handlers = [handler]
    logger.setLevel(level)
    logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Get an AIrsenal logger for the given module, e.g. ``__name__``."""
    return logging.getLogger(name)


configure_logging()


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
