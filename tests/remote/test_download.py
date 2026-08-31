"""Resumable downloads, and what happens when the connection never opens."""

import pytest
from curl_cffi import requests

from airsenal.remote.download import download_with_resume
from airsenal.remote.errors import RemoteError


class ExplodingSession:
    """A session whose `get` fails the way a refused connection does."""

    def __init__(self, failures: int, body: bytes = b"ok"):
        self.failures = failures
        self.body = body
        self.calls = 0

    def get(self, *_args, **_kwargs):
        self.calls += 1
        if self.calls <= self.failures:
            msg = "connection refused"
            raise requests.exceptions.ConnectionError(msg)
        return FakeResponse(self.body)


class FakeResponse:
    def __init__(self, body: bytes):
        self.body = body
        self.status_code = 200
        self.closed = False

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size=None):  # noqa: ARG002
        yield self.body

    def close(self):
        self.closed = True


@pytest.fixture
def session(monkeypatch):
    """Hand `download_with_resume` a session we control."""
    holder = {}

    def install(s):
        holder["session"] = s
        monkeypatch.setattr(requests, "Session", lambda *a, **k: s)
        return s

    return install


def test_a_connection_failure_is_retried_not_raised(tmp_path, session):
    """
    The request itself is inside the retry loop.

    It used to sit outside the `try`, so a refused connection escaped as a raw
    curl_cffi error on the first attempt - unretried, and past the
    `except RemoteError` every caller uses to fall back.
    """
    s = session(ExplodingSession(failures=2))
    dest = tmp_path / "file.csv"

    assert download_with_resume(url="https://example.invalid/f.csv", dest=dest) == dest
    assert s.calls == 3
    assert dest.read_bytes() == b"ok"


def test_every_attempt_failing_raises_remote_error(tmp_path, session):
    """A caller can name the failure in an `except` without importing the client."""
    s = session(ExplodingSession(failures=99))
    dest = tmp_path / "file.csv"

    with pytest.raises(RemoteError):
        download_with_resume(url="https://example.invalid/f.csv", dest=dest, attempts=3)
    assert s.calls == 3
