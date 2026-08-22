"""Lazily-created engine and sessions.

Nothing here runs at import: see tests/test_import_side_effects.py."""

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from airsenal.db.engine import get_connection_string
from airsenal.db.models.base import Base


# Engine and default session are created on first use, not at import. Creating them at
# import made importing any airsenal module open a database (and, via
# utils.NEXT_GAMEWEEK, call the FPL API), which is why the test suite could not be
# collected offline.
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

    Parameters
    ==========
    connection_string: str or None
        A SQLAlchemy connection string. If None, the string is resolved from the
        environment by `get_connection_string` on next use.
    """
    _db.reset(connection_string)


@contextmanager
def session_scope() -> Iterator[Session]:
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
