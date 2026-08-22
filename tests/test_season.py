from airsenal.framework.season import season_str_to_year, sort_seasons


def test_season_str_to_year():
    assert season_str_to_year("1819") == 2018


def test_sort_seasons():
    seasons = ["1819", "2021", "2122", "1920"]
    assert sort_seasons(seasons) == ["2122", "2021", "1920", "1819"]
    assert sort_seasons(seasons, desc=False) == ["1819", "1920", "2021", "2122"]
