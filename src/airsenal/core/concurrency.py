"""
Custom Queue class and counter, to allow us to get the length of a queue,
which in turn lets us do the tree-based optimization.

Taken from
https://gist.github.com/FanchenBao/d8577599c46eab1238a81857bb7277c9
by Fanchen Bao, based on this Stack Overflow thread:
https://stackoverflow.com/questions/41952413/get-length-of-queue-in-pythons-multiprocessing-library
"""

import faulthandler
import multiprocessing
import os
import threading
import time
from multiprocessing.queues import JoinableQueue
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from airsenal.core.env import AIRSENAL_HOME

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
    return AIRSENAL_HOME / "stalls"


def set_multiprocessing_start_method() -> None:
    """To fix change of default behaviour in multiprocessing on Python 3.8 and later
    on MacOS. Python 3.8 and later start processess using spawn by default, see:
    https://docs.python.org/3.8/library/multiprocessing.html#contexts-and-start-methods

    Idempotent, and forcing. `set_start_method` raises if a context has already
    been set, which used not to matter because nothing ran the pipeline twice in
    one process and nothing touched multiprocessing before this was called.
    Neither holds now: two AIrsenalPipeline.run() calls or a replay loop reach
    here more than once, and merely constructing a Queue anywhere first fixes the
    default context.

    Forcing is right rather than merely convenient. The transfer search hands its
    workers local progress callbacks, which pickle cannot serialise, so it can
    only run under fork; under spawn it does not run slower, it fails. Better to
    make that true here than to raise after a full prediction stage has run.
    """
    if os.name != "posix":
        return
    if multiprocessing.get_start_method(allow_none=True) == "fork":
        return
    multiprocessing.set_start_method("fork", force=True)


# The following implementation of custom MyQueue to avoid NotImplementedError
# when calling queue.qsize() in MacOS X comes almost entirely from this github
# discussion: https://github.com/keras-team/autokeras/issues/368
# Necessary modification is made to make the code compatible with Python3.


class SharedCounter:
    """
    A synchronized shared counter.
    The locking done by multiprocessing.Value ensures that only a single
    process or thread may read or write the in-memory ctypes object. However,
    in order to do n += 1, Python performs a read followed by a write, so a
    second process may read the old value before the new one is written by the
    first process. The solution is to use a multiprocessing.Lock to guarantee
    the atomicity of the modifications to Value.
    This class comes almost entirely from Eli Bendersky's blog:
    http://eli.thegreenplace.net/2012/01/04/shared-counter-with-pythons-multiprocessing/
    """

    def __init__(self, n: int = 0) -> None:
        self.count: Synchronized[int] = multiprocessing.Value("i", n)

    def increment(self, n: int = 1) -> None:
        """Increment the counter by n (default = 1)"""
        with self.count.get_lock():
            self.count.value += n

    @property
    def value(self) -> int:
        """Return the value of the counter"""
        return self.count.value


class CustomQueue(JoinableQueue[T]):
    """
    A portable implementation of multiprocessing.JoinableQueue.
    Because of multithreading / multiprocessing semantics, Queue.qsize() may
    raise the NotImplementedError exception on Unix platforms like Mac OS X
    where sem_getvalue() is not implemented. This subclass addresses this
    problem by using a synchronized shared counter (initialized to zero) and
    increasing / decreasing its value every time the put() and get() methods
    are called, respectively. This not only prevents NotImplementedError from
    being raised, but also allows us to implement a reliable version of both
    qsize() and empty().

    Subclassing JoinableQueue (rather than plain Queue) also gives us
    task_done()/join(), which track "unfinished tasks" via an internal
    semaphore rather than sem_getvalue(), so they work correctly on Mac OS X
    too. This lets callers detect when a dynamically-growing tree of tasks
    (where processing one task can enqueue more) is fully drained, without
    needing to independently know the expected total number of tasks.
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
        """Reliable implementation of multiprocessing.Queue.qsize()"""
        return self.size.value

    def empty(self) -> bool:
        """Reliable implementation of multiprocessing.Queue.empty()"""
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

    Parameters
    ----------
    name : str
        Used in the dump's filename, e.g. ``worker-3``.
    seconds : int or None
        How long one task may take before it counts as stalled. Defaults to
        ``AIRSENAL_STALL_SECONDS``, or `DEFAULT_STALL_SECONDS`.
    directory : Path or None
        Where to write dumps. Defaults to `stall_dump_dir()`.
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
