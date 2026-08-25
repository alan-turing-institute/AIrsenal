"""
Tests for the query caches.

The rule: no cache key may contain a database session. See
`airsenal/core/caching.py` for why.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from airsenal.core.caching import (
    cache_ignoring_session,
    clear_query_caches,
    registered_caches,
)
from airsenal.db.models import Base, Fixture
from airsenal.db.queries import predictions
from airsenal.db.queries.gameweeks import get_max_gameweek


@pytest.fixture
def sessions(tmp_path):
    """Two independent sessions onto the same database."""
    engine = create_engine(f"sqlite:///{tmp_path}/cache.db")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    clear_query_caches()
    with factory() as first, factory() as second:
        yield first, second
    clear_query_caches()


def _add_fixture(session, gameweek):
    session.add(
        Fixture(
            date="2025-08-10",
            gameweek=gameweek,
            home_team="AAA",
            away_team="BBB",
            season="9999",
            tag="test",
        )
    )
    session.commit()


def test_two_sessions_share_a_cached_answer(sessions):
    first, second = sessions
    _add_fixture(first, 20)

    assert get_max_gameweek("9999", dbsession=first) == 20
    # Keyed on the session, this would miss and re-query. It must hit.
    info_before = get_max_gameweek.cache_info()
    assert get_max_gameweek("9999", dbsession=second) == 20
    assert get_max_gameweek.cache_info().hits == info_before.hits + 1


def test_the_cache_actually_caches(sessions):
    first, _ = sessions
    _add_fixture(first, 20)

    get_max_gameweek("9999", dbsession=first)
    misses = get_max_gameweek.cache_info().misses
    for _ in range(10):
        get_max_gameweek("9999", dbsession=first)
    # This is the inner loop of the optimisation; losing the cache here turns a
    # two-minute run into a much longer one, with nothing failing.
    assert get_max_gameweek.cache_info().misses == misses


def test_different_seasons_are_cached_separately(sessions):
    """
    The old cache was `lru_cache(1)`, so a replay alternating between seasons
    evicted on every call and the cache did nothing at all.
    """
    first, _ = sessions
    _add_fixture(first, 20)
    session_two = first

    assert get_max_gameweek("9999", dbsession=first) == 20
    assert get_max_gameweek("8888", dbsession=session_two) == 38  # no fixtures
    misses = get_max_gameweek.cache_info().misses
    assert get_max_gameweek("9999", dbsession=first) == 20
    assert get_max_gameweek.cache_info().misses == misses


def test_clearing_invalidates(sessions):
    first, _ = sessions
    _add_fixture(first, 20)
    assert get_max_gameweek("9999", dbsession=first) == 20

    _add_fixture(first, 25)
    # still the old answer, which is the whole point of a cache
    assert get_max_gameweek("9999", dbsession=first) == 20

    clear_query_caches()
    assert get_max_gameweek("9999", dbsession=first) == 25


def test_every_cache_is_registered():
    assert list(registered_caches())


def test_the_decorator_passes_the_session_through_on_a_miss():
    seen = []

    @cache_ignoring_session(maxsize=4)
    def query(value, dbsession=None):
        seen.append(dbsession)
        return value * 2

    assert query(3, dbsession="A") == 6
    assert query(3, dbsession="B") == 6
    # one call, with the first caller's session
    assert seen == ["A"]


def test_a_player_id_is_not_looked_up_before_the_cache(monkeypatch):
    """
    Passing an int must not cost a database query.

    The optimiser calls this once per candidate player per candidate squad, so
    validating the id with `get_player` first roughly doubles the time of a
    transfer optimisation. Nothing fails if it comes back - it just gets slow.
    """

    def fail(*args, **kwargs):
        msg = "get_player must not be called for an integer player id"
        raise AssertionError(msg)

    def stub(player_id, tag, season, dbsession=None):
        return {1: float(player_id)}

    monkeypatch.setattr(predictions, "get_player", fail)
    monkeypatch.setattr(predictions, "_predicted_points_for_player_id", stub)
    assert predictions.get_predicted_points_for_player(7, "tag") == {1: 7.0}
