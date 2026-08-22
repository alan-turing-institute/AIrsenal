"""
The transfer search forks workers while a Rich progress bar is on screen.

Two things travel across that fork that must not: the console's lock, held by
the Live refresh thread whenever it is mid-render, and the database engine's
pool, which hands parent and children the same connection. Both failures are
silent - the worker stays alive, so nothing raises and the run simply stops,
with the progress bar frozen part-way.
"""

import multiprocessing
import os
import threading
import traceback

import pytest

from airsenal.core.console import _reset_console_after_fork, console
from airsenal.core.logging import configure_logging, get_logger
from airsenal.db.session import _db, _reset_engine_after_fork, get_session

pytestmark = [
    pytest.mark.skipif(not hasattr(os, "fork"), reason="fork is posix-only"),
    # Forking a multi-threaded process is exactly what is under test here, so
    # the interpreter's own warning about it is the expected output, not a bug.
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
]


def _log_in_child(queue) -> None:
    try:
        get_logger("airsenal.tests.fork").warning("hello from a forked child")
        queue.put("logged")
    except BaseException:
        queue.put(traceback.format_exc())


def test_forked_child_can_log_while_the_console_lock_is_held() -> None:
    """A child forked mid-render must not inherit a lock it can never acquire."""
    configure_logging()
    ctx = multiprocessing.get_context("fork")
    queue = ctx.Queue()

    holding = threading.Event()
    release = threading.Event()

    def hold_the_console_lock() -> None:
        with console._lock:
            holding.set()
            release.wait(30)

    threading.Thread(target=hold_the_console_lock, daemon=True).start()
    assert holding.wait(5), "could not take the console lock"

    child = ctx.Process(target=_log_in_child, args=(queue,))
    child.start()
    try:
        # Without the at-fork reset the child blocks forever inside
        # rich.console.Console.print, and this times out.
        assert queue.get(timeout=30) == "logged"
    finally:
        release.set()
        child.join(timeout=10)
        if child.is_alive():
            child.terminate()


def _connection_id_in_child(queue) -> None:
    session = get_session()
    queue.put(id(session.connection().connection.dbapi_connection))


def test_forked_child_does_not_share_the_parents_connection() -> None:
    """Parent and child must not issue statements down one inherited connection."""
    parent_connection = id(get_session().connection().connection.dbapi_connection)

    ctx = multiprocessing.get_context("fork")
    queue = ctx.Queue()
    child = ctx.Process(target=_connection_id_in_child, args=(queue,))
    child.start()
    child_connection = queue.get(timeout=30)
    child.join(timeout=10)

    assert child.exitcode == 0
    assert child_connection != parent_connection


def test_reset_console_after_fork_replaces_a_held_lock() -> None:
    console._lock.acquire()
    _reset_console_after_fork()
    assert console._lock.acquire(blocking=False), "lock was not replaced"
    console._lock.release()


def test_reset_engine_after_fork_drops_the_inherited_session() -> None:
    get_session()
    assert _db.default_session is not None
    _reset_engine_after_fork()
    assert _db.default_session is None
