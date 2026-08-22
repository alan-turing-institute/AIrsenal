"""
The one place in the package that posts to a Discord webhook.

The "is the URL well formed, post it, was the status 2xx" sequence was written
out three times, in two modules, with three slightly different log messages and
two different ideas of how to test whether a webhook was configured at all.
Having a single chokepoint also means a replay or a test cannot post to the real
channel by accident: there is exactly one call to make impossible.
"""

import re

from curl_cffi import requests

from airsenal.core.logging import get_logger
from airsenal.fetch.fpl_api import get_fetcher

logger = get_logger(__name__)

WEBHOOK_URL_PATTERN = re.compile(
    r"^.*(discord|discordapp)\.com/api/webhooks/(\d+)/([a-zA-Z0-9_-]+)$"
)


def get_webhook_url() -> str | None:
    """The configured webhook URL, or None if there isn't one."""
    return get_fetcher().DISCORD_WEBHOOK or None


def post_webhook(payload: dict, webhook_url: str | None = None) -> bool:
    """
    Post an embed to the configured Discord webhook.

    Returns whether anything was sent. A missing or malformed URL is a warning
    rather than an error - posting to Discord is optional, and a failure here
    must not lose the optimisation result that has just been computed.
    """
    webhook_url = webhook_url if webhook_url is not None else get_webhook_url()
    if not webhook_url:
        return False
    if not WEBHOOK_URL_PATTERN.match(webhook_url):
        logger.warning("Discord webhook url is malformed: %s", webhook_url)
        return False

    result = requests.post(webhook_url, json=payload)
    if 200 <= result.status_code < 300:
        logger.info("Discord webhook sent, status code: %s", result.status_code)
        return True
    logger.warning(
        "Discord webhook not sent, status code %s, response:\n%s",
        result.status_code,
        result.text,
    )
    return False
