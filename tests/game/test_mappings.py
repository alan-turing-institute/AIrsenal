import pytest

from airsenal.game.mappings import alternative_team_names, canonical_team_name


@pytest.mark.parametrize(
    ("name", "code"),
    [
        ("ARS", "ARS"),
        ("Arsenal FC", "ARS"),
        ("1", "ARS"),
        ("Spurs", "TOT"),
        ("Brighton & Hove Albion", "BHA"),
    ],
)
def test_a_code_or_any_alias_gives_the_code(name, code):
    assert canonical_team_name(name) == code


def test_an_unknown_name_gives_none():
    assert canonical_team_name("Real Madrid") is None


def test_no_alias_names_two_teams():
    """Otherwise which code an alias gives would depend on the table's order."""
    names = [*alternative_team_names]
    names += [alias for aliases in alternative_team_names.values() for alias in aliases]
    assert len(names) == len(set(names))
