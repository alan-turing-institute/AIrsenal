from airsenal.game.season import (
    CURRENT_SEASON,
    default_seasons,
    get_next_season,
    season_str_to_year,
    sort_seasons,
)


def test_season_str_to_year():
    assert season_str_to_year("1819") == 2018


def test_sort_seasons():
    seasons = ["1819", "2021", "2122", "1920"]
    assert sort_seasons(seasons) == ["2122", "2021", "1920", "1819"]
    assert sort_seasons(seasons, desc=False) == ["1819", "1920", "2021", "2122"]


def test_get_next_season_keeps_two_digits_per_year():
    assert get_next_season("1819") == "1920"
    assert get_next_season("0809") == "0910"


def test_default_seasons_are_this_one_and_the_three_before():
    seasons = default_seasons()
    assert seasons[0] == CURRENT_SEASON
    assert seasons == sort_seasons(seasons)
    assert len(seasons) == 4
