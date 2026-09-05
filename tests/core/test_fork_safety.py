"""
The transfer search forks workers while a Rich progress bar is on screen.

Two things that must not travel across that fork: the console's lock, held by
the Live refresh thread whenever it is mid-render, and the database engine's
pool, which hands parent and children the same connection. Both failures are
silent - the worker stays alive, so nothing raises and the run simply stops,
with the progress bar frozen part-way.
"""

import multiprocessing
import os
import subprocess
import sys
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


# Run out of process: the test needs to wedge sys.stdout, and pytest's capture
# replaces it with an object that has no BufferedWriter to wedge. Raw os.fork()
# rather than multiprocessing, because multiprocessing flushes the std streams
# before forking and would itself block on the lock under test.
_STDOUT_LOCK_PROBE = """
import os, sys, threading, time

if {fix}:
    from airsenal.core.console import _reset_console_after_fork

pipe_r, pipe_w = os.pipe()
result_r, result_w = os.pipe()
os.dup2(pipe_w, 1)
sys.stdout = os.fdopen(1, "w")

started = threading.Event()


def hog():
    started.set()
    sys.stdout.write("x" * (4 * 1024 * 1024))   # pipe fills: blocks, lock held
    sys.stdout.flush()


threading.Thread(target=hog, daemon=True).start()
started.wait(5)
time.sleep(1)

pid = os.fork()
if pid == 0:
    if {fix}:
        _reset_console_after_fork()
    sys.stdout.write("child\\n")
    sys.stdout.flush()
    os.write(result_w, b"wrote")
    os._exit(0)

# unblock the pipe only now, so a hung child can only be hung on the lock
threading.Thread(
    target=lambda: [None for _ in iter(lambda: os.read(pipe_r, 65536), b"")],
    daemon=True,
).start()

os.set_blocking(result_r, False)
answer = b""
deadline = time.time() + 20
while time.time() < deadline and not answer:
    try:
        answer = os.read(result_r, 64)
    except BlockingIOError:
        time.sleep(0.1)

os.write(2, answer or b"HUNG")
os.kill(pid, 9)
os._exit(0)
"""


def _run_stdout_lock_probe(*, fix: bool) -> str:
    finished = subprocess.run(
        [sys.executable, "-c", _STDOUT_LOCK_PROBE.format(fix=fix)],
        capture_output=True,
        timeout=90,
        check=False,
    )
    return finished.stderr.decode()[-16:]


def test_forked_child_can_write_while_stdout_is_being_written_to() -> None:
    """The lock below Rich's: CPython's own, which fork does not sanitise.

    Rich holds `console._lock` across the write to the terminal, and that write
    takes `sys.stdout`'s BufferedWriter lock. A child forked in that window
    inherits a locked buffer whose owner thread it does not have, and wedges on
    its first line of output.
    """
    assert _run_stdout_lock_probe(fix=True).endswith("wrote")


@pytest.mark.slow
def test_the_stdout_lock_probe_would_catch_a_regression() -> None:
    """Guard the guard: without the reset the same child hangs."""
    assert _run_stdout_lock_probe(fix=False).endswith("HUNG")
