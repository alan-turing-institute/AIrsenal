"""
test some db access helper functions
"""

import pytest

from ..utils import *


def test_get_player_name():
    """
    Should be able to find a player with integer argument
    """
    assert get_player_name(1) == "Petr Cech"


def test_get_player_id():
    """
    Should be able to find a player with string argument
    """
    assert get_player_id("Petr Cech") == 1
