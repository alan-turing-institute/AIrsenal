"""
The one place in the package that posts to a Discord webhook.

The "is the URL well formed, post it, was the status 2xx" sequence was written
out three times, in two modules, with three slightly different log messages and
two different ideas of how to test whether a webhook was configured at all.
Having a single chokepoint also means a replay or a test cannot post to the real
channel by accident: there is exactly one call to make impossible.

It lives beside the FPL client because Discord is the second host this package
talks to, and collecting every socket in one package is what makes "only
`remote` touches the network" a contract that can be enforced rather than a
convention that is nearly true. What gets said stays in `reporting`; this module
only knows how to say it.
"""

import re
from typing import Any

from curl_cffi import requests

from airsenal.core.env import get_env
from airsenal.core.logging import get_logger

logger = get_logger(__name__)

WEBHOOK_URL_PATTERN = re.compile(
    r"^.*(discord|discordapp)\.com/api/webhooks/(\d+)/([a-zA-Z0-9_-]+)$"
)


def get_webhook_url() -> str | None:
    """
    The configured webhook URL, or None if there isn't one.

    Read from the environment directly, rather than through the FPL API client,
    which copies this same value into `FPLDataFetcher.DISCORD_WEBHOOK`: posting
    to Discord has nothing to do with being logged in to FPL, and reaching for
    the client here would make an optional webhook depend on FPL credentials.
    """
    return get_env("DISCORD_WEBHOOK", str) or None


def post_webhook(payload: dict[str, Any], webhook_url: str | None = None) -> bool:
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

    try:
        result = requests.post(webhook_url, json=payload)
    except requests.exceptions.RequestException:
        # The post happens after the optimisation has finished, so an unreachable
        # Discord used to end the run and take the result with it - the one thing
        # this function's contract says it will not do.
        logger.warning("Discord webhook not sent, could not reach it", exc_info=True)
        return False
    if 200 <= result.status_code < 300:
        logger.info("Discord webhook sent, status code: %s", result.status_code)
        return True
    logger.warning(
        "Discord webhook not sent, status code %s, response:\n%s",
        result.status_code,
        result.text,
    )
    return False
