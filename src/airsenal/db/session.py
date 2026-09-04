"""Lazily-created engine and sessions. Nothing here runs at import."""

import os
from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from airsenal.core.caching import clear_query_caches
from airsenal.db.engine import get_connection_string
from airsenal.db.models import Base


# Engine and default session are created on first use, not at import, so that
# importing an airsenal module never opens a database or reaches the network.
class _DatabaseState:
    """Lazily-created engine and default session for the process."""

    def __init__(self) -> None:
        self.connection_string: str | None = None
        self.engine: Engine | None = None
        self.default_session: Session | None = None

    def reset(self, connection_string: str | None = None) -> None:
        if self.default_session is not None:
            self.default_session.close()
        if self.engine is not None:
            self.engine.dispose()
        self.connection_string = connection_string
        self.engine = None
        self.default_session = None


_db = _DatabaseState()


def _reset_engine_after_fork() -> None:
    """Stop a forked child from sharing the parent's database connection.

    `fork` copies the engine's pool, so parent and children end up issuing
    statements down one inherited connection - literally the same
    `sqlite3.Connection`, on the same file descriptor. SQLAlchemy's answer is
    `dispose(close=False)`: abandon the inherited connections (the parent still
    needs them, so do not close them) and let the child open its own on next
    use.

    The cached queries are deliberately left alone. They hold plain values
    rather than ORM objects, so they describe the database rather than the
    connection, and dropping them would make every worker re-read what the
    parent had already looked up.
    """
    if _db.engine is not None:
        _db.engine.dispose(close=False)
    _db.default_session = None


if hasattr(os, "register_at_fork"):  # pragma: no branch - posix only
    os.register_at_fork(after_in_child=_reset_engine_after_fork)


def get_engine() -> Engine:
    """
    The process-wide engine, created on first use.

    The tables are created if they do not exist yet, so callers do not have to know
    whether this is a fresh database.
    """
    if _db.engine is None:
        _db.engine = create_engine(_db.connection_string or get_connection_string())
        Base.metadata.create_all(_db.engine)
    return _db.engine


def create_session() -> Session:
    """Create a new session bound to the process-wide engine."""
    return sessionmaker(bind=get_engine(), autoflush=False)()


def get_session() -> Session:
    """
    The default session used throughout the package, created on first use.

    Prefer accepting a `dbsession` argument over calling this; it exists so that the
    `dbsession: Session | None = None` default can be resolved at call time rather
    than at import time.
    """
    if _db.default_session is None:
        _db.default_session = create_session()
    return _db.default_session


def configure_database(connection_string: str | None = None) -> None:
    """
    Point the package at a database, discarding any existing engine and session.

    Args:
        connection_string: A SQLAlchemy connection string. If None, the string is
            resolved from the environment by `get_connection_string` on next use.
    """
    _db.reset(connection_string)
    # cached query answers describe the database we were pointed at before
    clear_query_caches()


@contextmanager
def session_scope() -> Generator[Session]:
    """Provide a transactional scope around a series of operations."""
    dbsession = create_session()
    try:
        yield dbsession
        dbsession.commit()
    except Exception:
        dbsession.rollback()
        raise
    finally:
        dbsession.close()


def clean_database() -> None:
    """Drop every table and create them again, empty."""
    engine = get_engine()
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
