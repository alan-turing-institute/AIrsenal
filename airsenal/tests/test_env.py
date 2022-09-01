import os

import pytest

from airsenal.framework.env import delete_env, get_env, save_env


def test_get_env_none():
    assert get_env("FPL_LOGIN") is None


def test_get_env_default():
    assert get_env("FPL_LOGIN", default="TEST") == "TEST"


def test_get_env_value():
    os.environ["FPL_TEAM_ID"] = "123456"
    assert get_env("FPL_TEAM_ID") == 123456


def test_gen_env_invalid():
    with pytest.raises(KeyError):
        get_env("INVALID_KEY")


def test_save_env():
    save_env("FPL_LOGIN", "TEST")
    assert get_env("FPL_LOGIN") == "TEST"


def test_delete_env():
    assert get_env("FPL_LOGIN") is not None
    delete_env("FPL_LOGIN")
    assert get_env("FPL_LOGIN") is None
