"""Positions and chips."""

from airsenal.game.enums import Chip, Position


def test_position_compares_equal_to_its_string():
    """
    The enum has to keep behaving like a string.

    Query filters and string-keyed dicts compare against it directly, so `==`
    and indexing must work with a plain literal on the other side.
    """
    assert Position.GK == "GK"
    assert "GK" in {Position.GK: 1}


def test_position_formats_as_a_bare_string():
    """
    `str()` of a member is its value, not "Position.GK".

    A mixin enum without an explicit `__str__` formats as the latter, which
    would end up in log lines and database columns.
    """
    assert f"{Position.GK}" == "GK"
    assert str(Position.MID) == "MID"
    assert "{}".format(Position.DEF) == "DEF"  # noqa: UP032


def test_position_orderings_are_reverses_of_each_other():
    assert Position.back_to_front() == (
        Position.GK,
        Position.DEF,
        Position.MID,
        Position.FWD,
    )
    assert Position.front_to_back() == tuple(reversed(Position.back_to_front()))


def test_every_position_is_covered_by_both_orderings():
    assert set(Position.back_to_front()) == set(Position)
    assert set(Position.front_to_back()) == set(Position)


def test_chip_values_match_what_is_stored_in_the_database():
    """These strings are persisted in TransferSuggestion.chip_played."""
    assert {c.value for c in Chip} == {
        "wildcard",
        "free_hit",
        "bench_boost",
        "triple_captain",
    }


def test_only_wildcard_and_free_hit_rebuild_the_squad():
    assert Chip.WILDCARD.rebuilds_squad
    assert Chip.FREE_HIT.rebuilds_squad
    assert not Chip.BENCH_BOOST.rebuilds_squad
    assert not Chip.TRIPLE_CAPTAIN.rebuilds_squad
