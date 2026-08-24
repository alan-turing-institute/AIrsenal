"""Whole-database operations."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from airsenal.db.models import Base, Team
from airsenal.db.session import get_engine


def clean_database() -> None:
    """
    Clean up database
    """
    engine = get_engine()
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)


def database_is_empty(dbsession: Session) -> bool:
    """
    Basic check to determine whether the database is empty
    """
    return dbsession.scalars(select(Team).limit(1)).first() is None
