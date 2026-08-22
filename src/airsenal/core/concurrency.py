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

from airsenal.core.env import AIRSENAL_HOME

DEFAULT_STALL_SECONDS = 120
STALL_SECONDS_ENV = "AIRSENAL_STALL_SECONDS"


def stall_seconds() -> int:
    """How long one task may take before `StallWatchdog` calls it stalled."""
    raw = os.environ.get(STALL_SECONDS_ENV)
    return int(raw) if raw else DEFAULT_STALL_SECONDS


def stall_dump_dir() -> Path:
    """Where `StallWatchdog` writes its tracebacks."""
    return AIRSENAL_HOME / "stalls"


def set_multiprocessing_start_method():
    """To fix change of default behaviour in multiprocessing on Python 3.8 and later
    on MacOS. Python 3.8 and later start processess using spawn by default, see:
    https://docs.python.org/3.8/library/multiprocessing.html#contexts-and-start-methods

    Note that this should be called at most once, ideally protecteed within
    if __name__  == "__main__"
    """
    if os.name == "posix":
        multiprocessing.set_start_method("fork")


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

    def __init__(self, n=0):
        self.count = multiprocessing.Value("i", n)

    def increment(self, n=1):
        """Increment the counter by n (default = 1)"""
        with self.count.get_lock():
            self.count.value += n

    @property
    def value(self):
        """Return the value of the counter"""
        return self.count.value


class CustomQueue(JoinableQueue):
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

    def __init__(self):
        super().__init__(ctx=multiprocessing.get_context())
        self.size = SharedCounter(0)

    def put(self, *args, **kwargs):
        self.size.increment(1)
        super().put(*args, **kwargs)

    def get(self, *args, **kwargs):
        self.size.increment(-1)
        return super().get(*args, **kwargs)

    def qsize(self):
        """Reliable implementation of multiprocessing.Queue.qsize()"""
        return self.size.value

    def empty(self):
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
        self._last = time.monotonic()
        self._dumped = False
        self._lock = threading.Lock()

    def mark(self) -> None:
        """Record that progress has been made, i.e. a task started or finished."""
        with self._lock:
            self._last = time.monotonic()
            self._dumped = False

    def _stalled_for(self) -> float:
        with self._lock:
            return time.monotonic() - self._last

    def _should_dump(self) -> bool:
        with self._lock:
            if self._dumped or time.monotonic() - self._last < self.seconds:
                return False
            self._dumped = True
            return True

    def dump(self) -> Path:
        """Write every thread's traceback to a file, and return its path."""
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"stalled_{self.name}_{os.getpid()}.txt"
        with path.open("w") as dump_file:
            dump_file.write(
                f"{self.name} (pid {os.getpid()}) made no progress for "
                f"{self._stalled_for():.0f}s\n\n"
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
