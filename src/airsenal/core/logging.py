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
from multiprocessing import Queue, Value
from queue import Full
from typing import TYPE_CHECKING

from rich.logging import RichHandler

from airsenal.core.console import console

if TYPE_CHECKING:
    from multiprocessing.sharedctypes import Synchronized

_LOGGER_NAME = "airsenal"


class _RelayHandler(logging.handlers.QueueHandler):
    """Send records to the parent, dropping and counting them when it falls behind."""

    def __init__(
        self, queue: "Queue[logging.LogRecord | None]", dropped: "Synchronized[int]"
    ) -> None:
        super().__init__(queue)
        self.dropped = dropped

    def enqueue(self, record: logging.LogRecord) -> None:
        try:
            self.queue.put_nowait(record)
        except Full:
            with self.dropped.get_lock():
                self.dropped.value += 1


# Where a forked child should send its log records, and the count of those it
# had to drop, while this process owns a live display. Empty means "write them
# yourself", which is the normal case.
_relays: "list[tuple[Queue[logging.LogRecord | None], Synchronized[int]]]" = []


def configure_logging(level: int | str = logging.INFO) -> None:
    """Configure the AIrsenal logger to write coloured messages through Rich.

    At debug level each message is prefixed with the module and line that logged
    it, e.g. ``airsenal.prediction.run:112``.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(level)

    handler = RichHandler(
        console=console,
        show_time=False,
        show_level=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
    )
    debug = logger.getEffectiveLevel() <= logging.DEBUG
    fmt = "[dim]%(name)s:%(lineno)d[/dim]  %(message)s" if debug else "%(message)s"
    handler.setFormatter(logging.Formatter(fmt))

    logger.handlers = [handler]
    logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Get an AIrsenal logger for the given module, e.g. ``__name__``."""
    return logging.getLogger(name)


def _send_records_to_parent() -> None:
    """In a forked child, log to the relay queue instead of the terminal."""
    if not _relays:
        return
    logger = logging.getLogger(_LOGGER_NAME)
    logger.handlers = [_RelayHandler(*_relays[-1])]
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

    Only children forked inside the block are redirected. The queue is bounded
    (32767 records on macOS), so a child logging faster than the terminal can
    show drops records rather than blocking, and the parent says how many.
    """
    queue: Queue[logging.LogRecord | None] = Queue()
    dropped: Synchronized[int] = Value("i", 0)
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
    this_relay = (queue, dropped)
    _relays.append(this_relay)
    try:
        yield
    finally:
        _relays.remove(this_relay)
        queue.put(None)
        thread.join()
        if dropped.value:
            logging.getLogger(__name__).warning(
                "Dropped %s log messages from worker processes: they arrived "
                "faster than the terminal could show them.",
                dropped.value,
            )
