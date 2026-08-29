"""
Nothing reaches the FPL API without an explicit yes, or a flag saying so.

`apply/` is irreversible, so the confirmation is a safety feature and is tested
like one. It goes through `core.console.confirm` rather than a bare `input()`
precisely so these can be written.
"""

import pytest

from airsenal.apply import lineup as lineup_module
from airsenal.apply import transfers as transfers_module
from airsenal.core.console import confirm


@pytest.mark.parametrize("answer", ["y", "yes", "Y", "YES", " yes "])
def test_an_explicit_yes_proceeds(monkeypatch, answer):
    monkeypatch.setattr("builtins.input", lambda _: answer)
    assert confirm("go?", default=False) is True


@pytest.mark.parametrize("answer", ["", "n", "no", "maybe", "Ye", "1"])
def test_anything_else_does_not(monkeypatch, answer):
    """Irreversible means an unrecognised answer is a no, not a yes."""
    monkeypatch.setattr("builtins.input", lambda _: answer)
    assert confirm("go?", default=False) is False


def test_the_yes_biased_default_is_still_available(monkeypatch):
    """`airsenal run` asks whether to continue with a stale database; empty is yes."""
    monkeypatch.setattr("builtins.input", lambda _: "")
    assert confirm("continue?") is True
    assert confirm("continue?", default=True) is True


def test_transfers_are_not_applied_without_a_yes(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "no")
    assert transfers_module.check_proceed(1) is False


def test_a_hit_taking_transfer_is_confirmed_twice(monkeypatch):
    """More than two transfers means a points hit unless a chip is played."""
    answers = iter(["yes", "no"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    assert transfers_module.check_proceed(3) is False

    answers = iter(["yes", "yes"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    assert transfers_module.check_proceed(3) is True


def test_two_transfers_are_confirmed_once(monkeypatch):
    asked = []

    def record(prompt):
        asked.append(prompt)
        return "yes"

    monkeypatch.setattr("builtins.input", record)
    assert transfers_module.check_proceed(2) is True
    assert len(asked) == 1


def test_the_lineup_is_not_applied_without_a_yes(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "")
    monkeypatch.setattr(lineup_module, "formation_table", lambda *a, **k: "")
    assert lineup_module.check_proceed(squad=None, gameweek=1, tag="t") is False


def test_a_library_function_does_not_read_stdin_for_a_team_id():
    """
    It used to prompt, which made it uncallable from anything but a terminal.

    Raising names the fix instead.
    """

    class NoTeamId:
        FPL_TEAM_ID = None

    with pytest.raises(ValueError, match="No FPL team ID"):
        transfers_module.build_init_priced_transfers(fetcher=NoTeamId())
