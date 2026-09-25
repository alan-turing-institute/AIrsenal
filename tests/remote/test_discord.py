"""
Tests for the single Discord posting chokepoint.

What counts as a configured webhook, and what a failed post does instead of
raising.
"""

from unittest import mock

import pytest
from curl_cffi import requests

from airsenal.remote.discord import WEBHOOK_URL_PATTERN, post_webhook

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
    with mock.patch("airsenal.remote.discord.requests.post") as post:
        assert post_webhook({"a": 1}, url) is False
    post.assert_not_called()


def test_malformed_webhook_posts_nothing():
    # A malformed URL must not become a request to some other host.
    with mock.patch("airsenal.remote.discord.requests.post") as post:
        assert post_webhook({"a": 1}, "https://example.com/hook") is False
    post.assert_not_called()


def test_successful_post():
    with mock.patch("airsenal.remote.discord.requests.post") as post:
        post.return_value = mock.Mock(status_code=204)
        assert post_webhook({"a": 1}, VALID_URLS[0]) is True
    post.assert_called_once_with(VALID_URLS[0], json={"a": 1})


def test_failed_post_is_reported_but_does_not_raise():
    # Losing a Discord message must not lose the optimisation result with it.
    with mock.patch("airsenal.remote.discord.requests.post") as post:
        post.return_value = mock.Mock(status_code=500, text="boom")
        assert post_webhook({"a": 1}, VALID_URLS[0]) is False


def test_unreachable_discord_does_not_raise():
    # The post happens after the optimisation finished; an unreachable webhook must
    # not take the result down with it.
    with mock.patch("airsenal.remote.discord.requests.post") as post:
        post.side_effect = requests.exceptions.ConnectionError("down")
        assert post_webhook({"a": 1}, VALID_URLS[0]) is False
