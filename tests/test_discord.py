"""
Tests for the single Discord posting chokepoint.

Three copies of this logic used to exist, each with its own idea of what counted
as a configured webhook and what to log on failure. These pin the behaviour so
the copies cannot creep back.
"""

from unittest import mock

import pytest

from airsenal.reporting.discord import WEBHOOK_URL_PATTERN, post_webhook

VALID_URLS = [
    "https://discord.com/api/webhooks/123456789/abcDEF-ghi_jkl",
    "https://discordapp.com/api/webhooks/1/a",
    "https://ptb.discord.com/api/webhooks/999/xyz",
]
INVALID_URLS = [
    "https://example.com/api/webhooks/123/abc",
    "https://discord.com/api/webhooks/abc/def",  # id must be digits
    "https://discord.com/api/webhooks/123",  # no token
    "not a url at all",
]


@pytest.mark.parametrize("url", VALID_URLS)
def test_valid_webhook_urls_match(url):
    assert WEBHOOK_URL_PATTERN.match(url)


@pytest.mark.parametrize("url", INVALID_URLS)
def test_invalid_webhook_urls_do_not_match(url):
    assert WEBHOOK_URL_PATTERN.match(url) is None


@pytest.mark.parametrize("url", ["", None])
def test_no_webhook_configured_posts_nothing(url):
    with mock.patch("airsenal.reporting.discord.requests.post") as post:
        assert post_webhook({"a": 1}, url) is False
    post.assert_not_called()


def test_malformed_webhook_posts_nothing():
    # A malformed URL must not become a request to some other host.
    with mock.patch("airsenal.reporting.discord.requests.post") as post:
        assert post_webhook({"a": 1}, "https://example.com/hook") is False
    post.assert_not_called()


def test_successful_post():
    with mock.patch("airsenal.reporting.discord.requests.post") as post:
        post.return_value = mock.Mock(status_code=204)
        assert post_webhook({"a": 1}, VALID_URLS[0]) is True
    post.assert_called_once_with(VALID_URLS[0], json={"a": 1})


def test_failed_post_is_reported_but_does_not_raise():
    # Losing a Discord message must not lose the optimisation result with it.
    with mock.patch("airsenal.reporting.discord.requests.post") as post:
        post.return_value = mock.Mock(status_code=500, text="boom")
        assert post_webhook({"a": 1}, VALID_URLS[0]) is False
