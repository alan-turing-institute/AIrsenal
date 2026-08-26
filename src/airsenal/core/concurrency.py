"""
Process-level machinery for the multiprocess transfer search.

The start method the search needs, a queue whose length can be read on macOS,
and a watchdog that dumps stacks when a worker stops making progress.
"""

import faulthandler
import multiprocessing
import os
import threading
import time
from multiprocessing.queues import JoinableQueue
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from airsenal.core.env import airsenal_home

if TYPE_CHECKING:
    from multiprocessing.sharedctypes import Synchronized

DEFAULT_STALL_SECONDS = 120
STALL_SECONDS_ENV = "AIRSENAL_STALL_SECONDS"

# What `CustomQueue` carries. Deliberately a type parameter rather than the
# strategy tuple it holds in practice: core must not import from the optimizer.
T = TypeVar("T")


def stall_seconds() -> int:
    """How long one task may take before `StallWatchdog` calls it stalled."""
    raw = os.environ.get(STALL_SECONDS_ENV)
    return int(raw) if raw else DEFAULT_STALL_SECONDS


def stall_dump_dir() -> Path:
    """Where `StallWatchdog` writes its tracebacks."""
    return airsenal_home() / "stalls"


def set_multiprocessing_start_method() -> None:
    """
    Force the `fork` start method on posix. No-op elsewhere.

    macOS defaults to `spawn`, and the transfer search cannot run under it: it
    hands its workers local progress callbacks, which pickle cannot serialise.
    Under spawn it does not run slower, it fails.

    Idempotent and forcing, because `set_start_method` raises once a context has
    been set - and two `AIrsenalPipeline.run()` calls, a replay loop, or merely
    constructing a Queue first will each have set one. Better to force it here
    than to raise after a full prediction stage has run.
    """
    if os.name != "posix":
        return
    if multiprocessing.get_start_method(allow_none=True) == "fork":
        return
    multiprocessing.set_start_method("fork", force=True)


# The counter and queue below are adapted from
# https://github.com/keras-team/autokeras/issues/368 and
# https://gist.github.com/FanchenBao/d8577599c46eab1238a81857bb7277c9


class SharedCounter:
    """
    A counter several processes can increment.

    `multiprocessing.Value` makes a single read or write atomic, but `n += 1` is
    a read followed by a write, so a second process can read the old value before
    the first has written the new one. The lock covers the pair.

    From Eli Bendersky's blog:
    http://eli.thegreenplace.net/2012/01/04/shared-counter-with-pythons-multiprocessing/
    """

    def __init__(self, n: int = 0) -> None:
        self.count: Synchronized[int] = multiprocessing.Value("i", n)

    def increment(self, n: int = 1) -> None:
        """Increment the counter by n."""
        with self.count.get_lock():
            self.count.value += n

    @property
    def value(self) -> int:
        """The counter's current value."""
        return self.count.value


class CustomQueue(JoinableQueue[T]):
    """
    A `JoinableQueue` whose `qsize()` and `empty()` work on macOS.

    `Queue.qsize()` goes through `sem_getvalue()`, which macOS does not
    implement, so it raises NotImplementedError there. This keeps a
    `SharedCounter` instead, stepped on every `put()` and `get()`.

    `JoinableQueue` rather than plain `Queue` for `task_done()`/`join()`, which
    track unfinished tasks through a semaphore rather than `sem_getvalue()`. That
    is what lets the search tell when a tree of tasks that grows as it is walked -
    processing one node enqueues its children - has fully drained, without
    knowing the total in advance.
    """

    def __init__(self) -> None:
        super().__init__(ctx=multiprocessing.get_context())
        self.size = SharedCounter(0)

    def put(self, obj: T, block: bool = True, timeout: float | None = None) -> None:
        self.size.increment(1)
        super().put(obj, block, timeout)

    def get(self, block: bool = True, timeout: float | None = None) -> T:
        self.size.increment(-1)
        return super().get(block, timeout)

    def qsize(self) -> int:
        """Number of items on the queue, from the shared counter."""
        return self.size.value

    def empty(self) -> bool:
        """Whether the queue is empty, from the shared counter."""
        return not self.qsize()


class StallWatchdog:
    """
    Write this process's stacks to a file if it stops making progress.

    A worker that deadlocks - on a lock inherited across `fork`, say - stays
    alive, so the parent cannot tell it apart from one doing slow work: the run
    simply stops, with no error and no clue as to where. The watchdog notices
    that a single task has taken implausibly long and dumps the worker's own
    tracebacks, which is the only way to see where it stopped from the inside.

    Nothing is logged from the watchdog thread on purpose. If the process is
    wedged on the console lock, logging is exactly what would wedge the
    watchdog too; the file is written first and stands on its own.

    Only time spent *working* counts. A worker waiting on an empty queue has
    not stalled - that is most of them, most of the way through a run, and
    treating it as a stall would bury the one dump that matters.

    Args:
        name: Used in the dump's filename, e.g. `worker-3`.
        seconds: How long one task may take before it counts as stalled. Defaults
            to `AIRSENAL_STALL_SECONDS`, or `DEFAULT_STALL_SECONDS`.
        directory: Where to write dumps. Defaults to `stall_dump_dir()`.
    """

    def __init__(
        self,
        name: str,
        seconds: int | None = None,
        directory: Path | None = None,
    ) -> None:
        self.name = name
        self.seconds = seconds if seconds is not None else stall_seconds()
        self.directory = directory if directory is not None else stall_dump_dir()
        # None means idle - waiting for work rather than doing it.
        self._started: float | None = None
        self._dumped = False
        self._lock = threading.Lock()

    def busy(self) -> None:
        """Record that a task has just started, and start the clock."""
        with self._lock:
            self._started = time.monotonic()
            self._dumped = False

    def idle(self) -> None:
        """Record that there is nothing to do, and stop the clock."""
        with self._lock:
            self._started = None
            self._dumped = False

    def _stalled_for(self) -> float:
        with self._lock:
            if self._started is None:
                return 0.0
            return time.monotonic() - self._started

    def _should_dump(self) -> bool:
        with self._lock:
            if self._started is None or self._dumped:
                return False
            if time.monotonic() - self._started < self.seconds:
                return False
            self._dumped = True
            return True

    def dump(self) -> Path:
        """Write every thread's traceback to a file, and return its path."""
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"stalled_{self.name}_{os.getpid()}.txt"
        with path.open("w") as dump_file:
            dump_file.write(
                f"{self.name} (pid {os.getpid()}) spent {self._stalled_for():.0f}s "
                f"on a single task\n\n"
            )
            faulthandler.dump_traceback(file=dump_file, all_threads=True)
        return path

    def start(self) -> None:
        """Begin watching, on a daemon thread that dies with the process."""

        def watch() -> None:
            while True:
                time.sleep(min(5, max(1, self.seconds // 10)))
                if self._should_dump():
                    self.dump()

        threading.Thread(
            target=watch, daemon=True, name=f"{self.name}-watchdog"
        ).start()
