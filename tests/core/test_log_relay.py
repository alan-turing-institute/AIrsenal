"""
A forked worker must not write to a terminal the parent is drawing on.

The transfer search forks its workers while the progress bars are on screen. A
child that logs for itself lands in the middle of the display, which the parent
then redraws below the displaced frame - leaving a frozen copy of the whole
thing in the terminal. Records go back to the parent instead.
"""

import logging
import multiprocessing
import os
from collections.abc import Iterator

import pytest

from airsenal.core.logging import configure_logging, get_logger, relay_child_logs

pytestmark = [
    pytest.mark.skipif(not hasattr(os, "fork"), reason="fork is posix-only"),
    # Forking a multi-threaded process is what is under test here.
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
]

LOGGER_NAME = "airsenal.tests.relay"


class _Recorder(logging.Handler):
    """Collect the records this process emits, rather than printing them."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def recorder() -> Iterator[_Recorder]:
    """Replace the parent's Rich handler with one the test can read."""
    configure_logging()
    logger = logging.getLogger("airsenal")
    original = logger.handlers
    handler = _Recorder()
    logger.handlers = [handler]
    try:
        yield handler
    finally:
        logger.handlers = original


def _log_in_child(message: str) -> None:
    get_logger(LOGGER_NAME).warning(message)


def _report_handlers(queue) -> None:
    queue.put([type(h).__name__ for h in logging.getLogger("airsenal").handlers])


def _fork_and_wait(target, *args) -> None:
    child = multiprocessing.get_context("fork").Process(target=target, args=args)
    child.start()
    child.join(timeout=30)
    assert child.exitcode == 0


def test_a_child_forked_in_the_block_logs_through_the_parent(
    recorder: _Recorder,
) -> None:
    with relay_child_logs():
        _fork_and_wait(_log_in_child, "from the worker")

    assert [r.getMessage() for r in recorder.records] == ["from the worker"]


def test_the_child_does_not_write_the_record_itself() -> None:
    configure_logging()
    queue = multiprocessing.get_context("fork").Queue()

    with relay_child_logs():
        _fork_and_wait(_report_handlers, queue)
        assert queue.get(timeout=30) == ["QueueHandler"]


def test_a_child_forked_outside_the_block_keeps_its_own_handler() -> None:
    configure_logging()
    queue = multiprocessing.get_context("fork").Queue()

    _fork_and_wait(_report_handlers, queue)

    assert queue.get(timeout=30) == ["RichHandler"]


def test_the_same_message_from_several_workers_is_emitted_once(
    recorder: _Recorder,
) -> None:
    with relay_child_logs():
        for _ in range(3):
            _fork_and_wait(_log_in_child, "incomplete data in the db")
        _fork_and_wait(_log_in_child, "something else")

    assert [r.getMessage() for r in recorder.records] == [
        "incomplete data in the db",
        "something else",
    ]
