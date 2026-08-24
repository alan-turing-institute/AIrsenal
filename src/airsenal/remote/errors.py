"""
What a failed call to an external service raises.

Three modules - `pipeline/run.py`, `squad/state.py`, `squad/history.py` - used to
import `curl_cffi` for no reason other than to name its exceptions in an `except`
clause, which is how an HTTP client became a dependency of the optimiser. These
types let a caller say "the network failed, fall back to the database" without
knowing what the network is made of.
"""


class RemoteError(RuntimeError):
    """Something went wrong talking to an external service."""


class RemoteConnectionError(RemoteError):
    """The service could not be reached at all."""


class RemoteHTTPError(RemoteError):
    """
    The service answered, with a status that says no.

    Kept distinct from `RemoteConnectionError` because callers act on the
    difference: a 404 for one gameweek means "nothing there, keep looking", while
    an unreachable host means "stop, and assume the earliest gameweek".
    """

    def __init__(self, msg: str, status_code: int | None = None) -> None:
        super().__init__(msg)
        self.status_code = status_code
