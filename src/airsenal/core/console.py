"""Rich console, tables and progress bars: everything that renders."""

import io
import os
import sys
import threading
from collections.abc import Generator, Iterable, Iterator
from contextlib import contextmanager
from typing import Any, TextIO

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


def _fresh_stream(stream: TextIO) -> TextIO:
    """A new writer onto the same destination, with its own lock and buffer."""
    return io.TextIOWrapper(
        io.FileIO(os.dup(stream.fileno()), "w", closefd=True),
        encoding=stream.encoding,
        errors=stream.errors,
        line_buffering=True,
    )


def _reset_console_after_fork() -> None:
    """Give a forked child its own console, and its own way of writing.

    A child that forks while the terminal is being written to inherits two
    locks held by a thread that does not exist on its side of the fork, and
    either wedges it silently:

    1. `console._lock`, held by a Rich `Live` display for a whole refresh.
    2. The lock inside `sys.stdout`'s `BufferedWriter`, held for the write
       itself. CPython does not reset io locks across fork
       (python/cpython#50970).

    The live stack, render hooks and output buffer are dropped too: they belong
    to the parent's display, and the parent writes the buffered bytes itself.
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

    Bound to the shared `console`, not Rich's global default: two consoles give
    two uncoordinated Live stacks, which flicker against each other.
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
) -> Iterator[Any]:
    """Iterate over a sequence with a Rich progress bar."""
    with _new_progress() as progress:
        yield from progress.track(sequence, total=total, description=description)


@contextmanager
def progress_bar(*, transient: bool = False) -> Generator[Progress]:
    """Yield an AIrsenal-styled Rich Progress for tracking several tasks by hand."""
    with _new_progress(transient=transient) as progress:
        yield progress


def confirm(question: str, default: bool = True) -> bool:
    """
    Ask a yes/no question at the terminal.

    Args:
        default: What an empty answer, or anything unrecognised, means.
    """
    suffix = "[Y/n]" if default else "[y/N]"
    answer = input(f"{question} {suffix} ").strip().lower()
    if default:
        return answer not in ("n", "no")
    return answer in ("y", "yes")
