import logging
import os

import pytest

from airsenal.cli.diagnostics import print_env
from airsenal.core.env import (
    AIRSENAL_ENV_KEYS,
    SECRET_ENV_KEYS,
    delete_env,
    get_env,
    save_env,
)


@pytest.fixture
def unset_login():
    """FPL_LOGIN back to however this machine had it, whatever the test did."""
    original = get_env("FPL_LOGIN", str)
    delete_env("FPL_LOGIN")
    yield
    delete_env("FPL_LOGIN")
    if original is not None:
        save_env("FPL_LOGIN", original)


def test_an_unset_setting_is_none(unset_login):
    assert get_env("FPL_LOGIN", str) is None


def test_a_setting_is_converted_to_the_type_asked_for():
    os.environ["FPL_TEAM_ID"] = "123456"
    assert get_env("FPL_TEAM_ID", int) == 123456


def test_an_unrecognised_key_is_refused():
    with pytest.raises(KeyError):
        get_env("INVALID_KEY", str)


def test_a_saved_setting_is_read_back(unset_login):
    save_env("FPL_LOGIN", "TEST")
    assert get_env("FPL_LOGIN", str) == "TEST"


def test_a_deleted_setting_is_gone(unset_login):
    save_env("FPL_LOGIN", "TEST")
    delete_env("FPL_LOGIN")
    assert get_env("FPL_LOGIN", str) is None


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


def test_every_secret_key_is_a_known_key():
    """A typo in SECRET_ENV_KEYS would silently redact nothing."""
    assert set(AIRSENAL_ENV_KEYS) >= SECRET_ENV_KEYS
