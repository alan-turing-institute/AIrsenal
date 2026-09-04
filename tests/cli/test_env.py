import logging
import os

from airsenal.cli.env import print_env


def test_print_env_does_not_echo_secrets(caplog):
    """The bulk dump reports a credential as set, never its value."""
    os.environ["FPL_PASSWORD"] = "hunter2"
    os.environ["DISCORD_WEBHOOK"] = "https://discord.com/api/webhooks/secret"
    os.environ["FPL_TEAM_ID"] = "123456"
    try:
        with caplog.at_level(logging.INFO):
            print_env()
    finally:
        del os.environ["FPL_PASSWORD"]
        del os.environ["DISCORD_WEBHOOK"]

    output = caplog.text
    assert "hunter2" not in output
    assert "webhooks/secret" not in output
    assert "FPL_PASSWORD: ***" in output
    assert "DISCORD_WEBHOOK: ***" in output
    # a plain setting is still shown
    assert "FPL_TEAM_ID: 123456" in output
