"""Building the player summary files from a season's FPL dump."""

from airsenal.export.player_summary import opta_code


def test_the_api_opta_code_is_kept():
    assert opta_code({"code": 1, "opta_code": "man2"}) == "man2"


def test_an_older_season_builds_it_from_the_code():
    """Before 24/25 the dump has no `opta_code`, only the `code` it is made from."""
    assert opta_code({"code": 451340}) == "p451340"
