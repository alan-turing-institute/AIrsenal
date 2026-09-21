"""
Tests for the pipeline not hanging or destroying data.
"""

import sqlite3
import time
from multiprocessing import Event, Process

import pytest

from airsenal.framework import schema
from airsenal.scripts import fill_transfersuggestion_table as fts


def _exit_ok():
    return


def _exit_crash():
    msg = "worker blew up"
    raise RuntimeError(msg)


def _sleep_forever():
    while True:
        time.sleep(0.1)


@pytest.fixture
def empty_output_dir(tmp_path, monkeypatch):
    """is_finished() counts files in OUTPUT_DIR, so point it somewhere empty."""
    out = tmp_path / "airsopt"
    out.mkdir()
    monkeypatch.setattr(fts, "OUTPUT_DIR", str(out))
    return out


def test_wait_for_processes_raises_when_a_worker_dies(empty_output_dir):
    """A crashed worker must abort the run rather than leaving the others to wait
    forever for output that will never be produced."""
    procs = [Process(target=_exit_crash), Process(target=_sleep_forever)]
    for p in procs:
        p.daemon = True
        p.start()

    abort = Event()
    with pytest.raises(RuntimeError, match="strategy workers failed"):
        fts.wait_for_processes(procs, abort, num_expected_outputs=5, poll_seconds=0.05)

    assert abort.is_set()
    assert not any(p.is_alive() for p in procs)


def test_wait_for_processes_raises_on_incomplete_output(empty_output_dir):
    """Workers exiting cleanly without producing everything is also a failure, not
    a silently truncated result set."""
    procs = [Process(target=_exit_ok)]
    for p in procs:
        p.start()

    with pytest.raises(RuntimeError, match="expected results"):
        fts.wait_for_processes(
            procs, Event(), num_expected_outputs=3, poll_seconds=0.05
        )


def test_wait_for_processes_returns_when_output_complete(empty_output_dir):
    procs = [Process(target=_exit_ok)]
    for p in procs:
        p.start()
    (empty_output_dir / "strategy_a.json").write_text("{}")

    fts.wait_for_processes(procs, Event(), num_expected_outputs=1, poll_seconds=0.05)


def test_is_finished_tolerates_extra_files(empty_output_dir):
    """Overshooting the expected count must still count as finished, or the workers
    never stop."""
    for i in range(3):
        (empty_output_dir / f"strategy_{i}.json").write_text("{}")
    assert fts.is_finished(2) is True


def test_backup_db_copies_and_prunes(tmp_path, monkeypatch):
    db_file = tmp_path / "data.db"
    conn = sqlite3.connect(db_file)
    conn.execute("CREATE TABLE t (x int)")
    conn.execute("INSERT INTO t VALUES (42)")
    conn.commit()
    conn.close()

    monkeypatch.setattr(schema, "get_db_file", lambda: db_file)

    written = []
    for _ in range(3):
        path = schema.backup_db(keep=2)
        assert path is not None
        assert path.exists(), "backup written"
        written.append(path)

    assert len(set(written)) == 3, "backups in the same second must not collide"

    backups = sorted(tmp_path.glob("data.db.backup-*"))
    assert len(backups) == 2, "should keep only the most recent backups"
    assert not written[0].exists(), "oldest backup should have been pruned"

    # the backup is a usable copy, not an empty file
    conn = sqlite3.connect(backups[-1])
    assert conn.execute("SELECT x FROM t").fetchone()[0] == 42
    conn.close()


def test_backup_db_no_database(tmp_path, monkeypatch):
    """Nothing to back up is not an error."""
    monkeypatch.setattr(schema, "get_db_file", lambda: tmp_path / "missing.db")
    assert schema.backup_db() is None


def test_backup_db_skips_postgres(monkeypatch):
    monkeypatch.setattr(schema, "get_db_file", lambda: None)
    assert schema.backup_db() is None
