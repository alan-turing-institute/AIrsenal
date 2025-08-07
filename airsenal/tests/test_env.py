import os

import pytest

from airsenal.framework.env import delete_env, get_env, save_env


def test_get_env_none():
    assert get_env("FPL_LOGIN", str) is None


def test_get_env_value():
    os.environ["FPL_TEAM_ID"] = "123456"
    assert get_env("FPL_TEAM_ID", int) == 123456


def test_gen_env_invalid():
    with pytest.raises(KeyError):
        get_env("INVALID_KEY", str)


def test_save_env():
    save_env("FPL_LOGIN", "TEST")
    assert get_env("FPL_LOGIN", str) == "TEST"


def test_delete_env():
    orig = get_env("FPL_LOGIN", str)
    assert orig is not None
    delete_env("FPL_LOGIN")
    assert get_env("FPL_LOGIN", str) is None
    save_env("FPL_LOGIN", orig)
    assert get_env("FPL_LOGIN", str) == orig
