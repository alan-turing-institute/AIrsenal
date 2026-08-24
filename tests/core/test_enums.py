"""Positions and chips."""

from airsenal.core.enums import Chip, Position


def test_position_compares_equal_to_its_string():
    """
    Existing code filters SQLAlchemy queries and indexes dicts with plain strings,
    so the enum has to keep behaving like one while those are migrated.
    """
    assert Position.GK == "GK"
    assert "GK" in {Position.GK: 1}


def test_position_formats_as_a_bare_string():
    """
    Regression guard for the 3.11 mixin-enum change. Without an explicit __str__,
    f"{Position.GK}" becomes "Position.GK", which would end up in log lines and
    database columns.
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
