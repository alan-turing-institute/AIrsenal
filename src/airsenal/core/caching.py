"""
The query caches, kept together so they can all be dropped at once.

The rule these exist to support: **no `lru_cache` on a function that takes a
`Session`**. A Session hashes by identity, so the cache key would silently
include which session object asked, and none of the resulting stale or missed
answers fail loudly.

Cached queries therefore key on the values that determine the answer - a player
id, a date, a season - and register themselves here so that anything which
invalidates them can say so.
"""

from collections.abc import Callable, Iterator
from contextvars import ContextVar
from functools import lru_cache, wraps
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

P = ParamSpec("P")
R = TypeVar("R")

_caches: list[Callable[[], None]] = []

# The session for the call currently in flight. A ContextVar rather than a
# plain global so that threads - the progress consumer in the transfer search,
# for one - cannot see each other's.
_current_session: ContextVar["Session | None"] = ContextVar(
    "airsenal_query_session", default=None
)


def cache_ignoring_session(
    maxsize: int | None = 128,
) -> Callable[[Callable[..., R]], Callable[..., R]]:
    """
    Cache a query function on its arguments, but never on its `dbsession`.

    The session still reaches the function on a cache miss - it is passed out of
    band rather than as part of the key - so a caller can still choose which
    database is read. Every other argument must be hashable, which in practice
    means passing a player id rather than a Player.
    """

    def decorate(func: Callable[..., R]) -> Callable[..., R]:
        @lru_cache(maxsize=maxsize)
        def cached(*args: Any, **kwargs: Any) -> R:
            return func(*args, dbsession=_current_session.get(), **kwargs)

        @wraps(func)
        def wrapper(*args: Any, dbsession: "Session | None" = None, **kwargs: Any) -> R:
            token = _current_session.set(dbsession)
            try:
                return cached(*args, **kwargs)
            finally:
                _current_session.reset(token)

        wrapper.cache_clear = cached.cache_clear  # type: ignore[attr-defined]
        wrapper.cache_info = cached.cache_info  # type: ignore[attr-defined]
        register_cache(cached)
        return wrapper

    return decorate


def register_cache(cached: object) -> None:
    """Register an `lru_cache`-wrapped function so it can be cleared."""
    clear = getattr(cached, "cache_clear", None)
    if clear is None:
        msg = f"{cached!r} has no cache_clear; is it wrapped in lru_cache?"
        raise TypeError(msg)
    _caches.append(clear)


def clear_query_caches() -> None:
    """
    Drop every cached query answer.

    Call this after writing anything the cached queries read - filling the
    fixture table, or pointing the package at a different database.
    """
    for clear in _caches:
        clear()


def registered_caches() -> Iterator[Callable[[], None]]:
    """The registered cache-clearing callables, for tests and diagnostics."""
    yield from _caches
