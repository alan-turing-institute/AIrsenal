"""Logging setup.

Kept apart from console.py so that code which only needs a logger does not pull
in Rich.
"""

import logging
import logging.handlers
import os
import threading
from collections.abc import Generator
from contextlib import contextmanager
from multiprocessing import Queue

from rich.logging import RichHandler

from airsenal.core.console import console

_LOGGER_NAME = "airsenal"

# Where a forked child should send its log records, while this process owns a
# live display. Empty means "write them yourself", which is the normal case.
_relay_queues: "list[Queue[logging.LogRecord | None]]" = []


def configure_logging(level: int | str = logging.INFO) -> None:
    """Configure the AIrsenal logger to write bare, coloured messages through Rich."""
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
def relay_child_logs() -> Generator[None]:
    """Have children forked in this block log through the parent, not directly.

    A child writing to the terminal itself leaves frozen copies of the parent's
    live display (the transfer search's progress bars) on screen. Relayed
    records are emitted by the parent above the display instead, and identical
    messages only once.

    Only children forked inside the block are redirected.
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
