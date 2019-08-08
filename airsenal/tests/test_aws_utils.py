"""
test the functions that download from S3, and construct return strings.
"""

import pytest
from ..framework.aws_utils import *

@pytest.mark.skip("No league standings before start of season")
def test_get_league_standings_string():
    s = get_league_standings_string()
    assert not s.startswith("Problem")
