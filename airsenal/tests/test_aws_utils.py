"""
test the functions that download from S3, and construct return strings.
"""

from ..framework.aws_utils import *


def test_get_league_standings_string():
    s = get_league_standings_string()
    assert not s.startswith("Problem")
