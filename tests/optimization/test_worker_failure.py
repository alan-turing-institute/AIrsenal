"""
A worker that dies must not leave the parent waiting forever.

JoinableQueue.join() only watches its unfinished-task counter, so a worker that
raises before calling task_done() blocks the parent indefinitely. That is what turns
any crash in the strategy search into a silent hang, with the progress bar stopped
part-way and nothing to say why.
"""

import multiprocessing
import threading

import pytest

from airsenal.core.concurrency import CustomQueue
from airsenal.optimization.run_transfers import _wait_for_queue


def _die_without_finishing(queue):
    queue.get()
    raise SystemExit(1)


def _finish_properly(queue):
    queue.get()
    queue.task_done()


@pytest.fixture
def mp_context():
    return multiprocessing.get_context("fork")


def test_returns_once_every_task_is_done(mp_context):
    queue = CustomQueue()
    proc = mp_context.Process(target=_finish_properly, args=(queue,), daemon=True)
    proc.start()
    queue.put("task")
    try:
        _wait_for_queue(queue, [proc])
    finally:
        proc.join(timeout=5)


def test_raises_instead_of_hanging_when_a_worker_dies(mp_context):
    queue = CustomQueue()
    proc = mp_context.Process(target=_die_without_finishing, args=(queue,), daemon=True)
    proc.start()
    queue.put("task")
    try:
        with pytest.raises(RuntimeError, match="worker 0 exited with 1"):
            _wait_for_queue(queue, [proc])
    finally:
        proc.join(timeout=5)


def test_the_error_says_how_to_get_the_real_traceback(mp_context):
    queue = CustomQueue()
    proc = mp_context.Process(target=_die_without_finishing, args=(queue,), daemon=True)
    proc.start()
    queue.put("task")
    try:
        with pytest.raises(RuntimeError, match="--num-thread 1"):
            _wait_for_queue(queue, [proc])
    finally:
        proc.join(timeout=5)


@pytest.mark.slow  # deliberately waits on a join that never returns
def test_a_bare_join_would_hang(mp_context):
    """
    Documents why the helper exists: the obvious queue.join() never returns here.
    """
    queue = CustomQueue()
    proc = mp_context.Process(target=_die_without_finishing, args=(queue,), daemon=True)
    proc.start()
    queue.put("task")
    joiner = threading.Thread(target=queue.join, daemon=True)
    joiner.start()
    joiner.join(timeout=5)
    try:
        assert joiner.is_alive(), "expected queue.join() to still be blocked"
    finally:
        proc.join(timeout=5)
