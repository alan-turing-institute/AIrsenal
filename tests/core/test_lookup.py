import pytest

from airsenal.core.lookup import ConfigError, lookup

TABLE = {"basic": object, "other": object}


def test_lookup_returns_the_registered_entry():
    assert lookup(TABLE, "basic", "dummy model") is object


def test_lookup_reports_an_unknown_name_with_the_valid_ones():
    with pytest.raises(ConfigError, match=r"Unknown dummy model 'nope'.*basic, other"):
        lookup(TABLE, "nope", "dummy model")
