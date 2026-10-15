"""Which chips a season gives twice, and which are used up by a given gameweek."""

import pytest

from airsenal.game.chips import chips_used_up, comes_twice, season_half
from airsenal.game.enums import Chip


@pytest.mark.parametrize(("gameweek", "half"), [(1, 1), (19, 1), (20, 2), (38, 2)])
def test_the_halves_split_after_gameweek_19(gameweek, half):
    assert season_half(gameweek) == half


@pytest.mark.parametrize("season", ["1819", "2425", "2526", "2627"])
def test_every_season_has_a_wildcard_for_each_half(season):
    assert comes_twice(Chip.WILDCARD, season)


@pytest.mark.parametrize("chip", [Chip.FREE_HIT, Chip.BENCH_BOOST, Chip.TRIPLE_CAPTAIN])
@pytest.mark.parametrize(("season", "twice"), [("2425", False), ("2526", True)])
def test_every_chip_comes_twice_from_2025_26(chip, season, twice):
    assert comes_twice(chip, season) is twice


def test_a_chip_played_in_the_first_half_is_back_in_the_second():
    played = [(18, Chip.WILDCARD), (19, Chip.BENCH_BOOST)]

    assert chips_used_up(played, 19, "2526") == (Chip.WILDCARD, Chip.BENCH_BOOST)
    assert chips_used_up(played, 20, "2526") == ()


def test_before_2025_26_only_the_wildcard_comes_back():
    played = [(18, Chip.WILDCARD), (19, Chip.BENCH_BOOST)]

    assert chips_used_up(played, 20, "2425") == (Chip.BENCH_BOOST,)


def test_a_gameweek_without_a_chip_uses_nothing_up():
    assert chips_used_up([(5, None), (6, None)], 7, "2526") == ()
