"""Logging setup.

Kept apart from console.py so that code which only needs a logger does not pull
in Rich.
"""

import logging
import logging.handlers
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from multiprocessing import Queue

from rich.logging import RichHandler

from airsenal.core.console import console

_LOGGER_NAME = "airsenal"

# Where a forked child should send its log records, while this process owns a
# live display. Empty means "write them yourself", which is the normal case. A
# one-element stack rather than a plain name, because a child reads it after the
# fork and rebinding a module-level name from inside a function needs `global`.
_relay_queues: "list[Queue[logging.LogRecord | None]]" = []


def configure_logging(level: int | str = logging.INFO) -> None:
    """Configure the AIrsenal logger to write through Rich.

    Designed for a CLI user reading a terminal, not an operator reading a log
    file: no timestamps, logger names, or file paths - just the message,
    colour-coded by level, with Rich markup in the message rendered.
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


def _send_records_to_parent() -> None:
    """In a forked child, log to the relay queue instead of the terminal."""
    if not _relay_queues:
        return
    logger = logging.getLogger(_LOGGER_NAME)
    logger.handlers = [logging.handlers.QueueHandler(_relay_queues[-1])]
    logger.propagate = False


if hasattr(os, "register_at_fork"):  # pragma: no branch - posix only
    os.register_at_fork(after_in_child=_send_records_to_parent)


@contextmanager
def relay_child_logs() -> Iterator[None]:
    """Have children forked in this block log through the parent, not directly.

    A child that writes to the terminal itself lands in the middle of whatever
    live display the parent is running - the transfer search's progress bars,
    say. The child knows nothing about that display, so the frame its message
    displaces is never erased: it stays on screen, and the parent redraws the
    bars below it. One warning per worker is enough to litter the terminal with
    frozen copies of the whole display.

    Records put on the queue are emitted here instead, through the handler that
    knows how to print above a live display. Identical messages are emitted
    once: every worker is looking at the same database, so a condition worth
    warning about is one all of them hit, and one copy is the news.

    Only children forked *inside* the block are redirected, and only on
    platforms that fork - which is how the search starts its workers.
    """
    queue: Queue[logging.LogRecord | None] = Queue()
    seen: set[tuple[int, str]] = set()

    def relay() -> None:
        while True:
            record = queue.get()
            if record is None:
                break
            key = (record.levelno, record.getMessage())
            if key in seen:
                continue
            seen.add(key)
            logging.getLogger(record.name).handle(record)

    thread = threading.Thread(target=relay, daemon=True)
    thread.start()
    _relay_queues.append(queue)
    try:
        yield
    finally:
        _relay_queues.remove(queue)
        queue.put(None)
        thread.join()
