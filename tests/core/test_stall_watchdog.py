"""
The stall watchdog, which is the only thing that says where a run stopped.

A wedged worker stays alive, so the parent sees nothing wrong and the run simply
stops making progress.
"""

import threading
import time

from airsenal.core.concurrency import (
    DEFAULT_STALL_SECONDS,
    STALL_SECONDS_ENV,
    StallWatchdog,
    stall_dump_dir,
    stall_seconds,
)


def test_dump_names_the_worker_and_shows_its_stack(tmp_path) -> None:
    watchdog = StallWatchdog("worker-7", seconds=1, directory=tmp_path / "stalls")
    path = watchdog.dump()

    text = path.read_text()
    assert "worker-7" in path.name
    assert "worker-7" in text
    assert "on a single task" in text
    # the dump is a traceback, and this test is somewhere in it
    assert "test_dump_names_the_worker_and_shows_its_stack" in text


def test_dumps_once_a_task_takes_too_long(tmp_path) -> None:
    directory = tmp_path / "stalls"
    watchdog = StallWatchdog("worker-0", seconds=1, directory=directory)
    watchdog.start()
    watchdog.busy()

    deadline = time.time() + 15
    while not list(directory.glob("*.txt")) and time.time() < deadline:
        time.sleep(0.2)

    assert list(directory.glob("*.txt")), "watchdog never dumped"


def test_stays_quiet_while_tasks_keep_finishing(tmp_path) -> None:
    directory = tmp_path / "stalls"
    watchdog = StallWatchdog("worker-1", seconds=2, directory=directory)
    watchdog.start()

    for _ in range(12):
        watchdog.busy()
        time.sleep(0.25)

    assert not directory.exists() or not list(directory.glob("*.txt"))


def test_dumps_again_only_after_the_next_task(tmp_path) -> None:
    """One dump per stalled task, not one every five seconds."""
    watchdog = StallWatchdog("worker-2", seconds=0, directory=tmp_path / "stalls")
    watchdog.busy()
    assert watchdog._should_dump()
    assert not watchdog._should_dump()
    watchdog.busy()
    assert watchdog._should_dump()


def test_waiting_for_work_is_not_a_stall(tmp_path) -> None:
    """Most workers are idle at the tail of a run; that is not a hang.

    Counting it as one buried the single dump that mattered under nine that
    did not.
    """
    directory = tmp_path / "stalls"
    watchdog = StallWatchdog("worker-4", seconds=0, directory=directory)
    watchdog.idle()
    assert not watchdog._should_dump()

    watchdog.start()
    time.sleep(2)
    assert not directory.exists() or not list(directory.glob("*.txt"))


def test_threshold_is_configurable(monkeypatch) -> None:
    monkeypatch.delenv(STALL_SECONDS_ENV, raising=False)
    assert stall_seconds() == DEFAULT_STALL_SECONDS
    monkeypatch.setenv(STALL_SECONDS_ENV, "7")
    assert stall_seconds() == 7


def test_dumps_land_under_airsenal_home() -> None:
    assert stall_dump_dir().name == "stalls"


def test_marking_is_safe_from_another_thread(tmp_path) -> None:
    watchdog = StallWatchdog("worker-3", seconds=60, directory=tmp_path)
    threads = [threading.Thread(target=watchdog.busy) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert watchdog._stalled_for() < 1
