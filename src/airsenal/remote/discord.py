"""Posting to a Discord webhook."""

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
    """The configured webhook URL, or None if there isn't one."""
    return get_env("DISCORD_WEBHOOK", str) or None


def post_webhook(payload: dict[str, Any], webhook_url: str | None = None) -> bool:
    """
    Post an embed to the configured Discord webhook.

    Returns whether anything was sent.
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
