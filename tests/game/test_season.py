from airsenal.game.season import (
    FIRST_SEASON_WITH_EXPECTED_GOALS,
    has_expected_goals,
    season_str_to_year,
    sort_seasons,
)


def test_season_str_to_year():
    assert season_str_to_year("1819") == 2018


def test_sort_seasons():
    seasons = ["1819", "2021", "2122", "1920"]
    assert sort_seasons(seasons) == ["2122", "2021", "1920", "1819"]
    assert sort_seasons(seasons, desc=False) == ["1819", "1920", "2021", "2122"]


def test_expected_goals_start_in_2223():
    """
    Which season the FPL API started reporting expected goals for.

    A fact about what FPL publishes, not about any model, so the two models
    fitted to expected goals read it from here rather than each naming the
    season in their own error message.
    """
    assert FIRST_SEASON_WITH_EXPECTED_GOALS == "2223"
    assert not has_expected_goals("2122")
    assert has_expected_goals("2223")
    assert has_expected_goals("2526")
