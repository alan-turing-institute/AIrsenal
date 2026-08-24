"""
Every swappable component, checked the same way.

There are five kinds of pluggable component, and each keeps a plain dict of name
to zero-argument factory in its own package's `__init__.py`. Adding an
implementation means adding one entry, and that entry is covered here
automatically: it must build with no arguments, provide the method its protocol
names, and - for the four kinds a flag selects - be reachable by that name from
the command line.
"""

import pytest
from typer.testing import CliRunner

from airsenal.cli.main import app
from airsenal.optimization.squad_optimizers import SQUAD_OPTIMIZERS
from airsenal.optimization.strategies import TRANSFER_STRATEGIES
from airsenal.optimization.transfer_optimizers import TRANSFER_OPTIMIZERS
from airsenal.prediction.player_models import (
    PLAYER_MODELS,
)
from airsenal.prediction.team_models import (
    TEAM_MODELS,
)

TABLES = {
    "player model": (PLAYER_MODELS, ("fit", "get_probs")),
    "team model": (TEAM_MODELS, ("fit", "add_new_team", "predict_score_n_proba")),
    "transfer strategy": (TRANSFER_STRATEGIES, ("propose",)),
    "squad optimizer": (SQUAD_OPTIMIZERS, ("optimize",)),
    "transfer optimizer": (TRANSFER_OPTIMIZERS, ("search",)),
}

ENTRIES = [(kind, name) for kind, (table, _) in TABLES.items() for name in table]


@pytest.mark.parametrize(("kind", "name"), ENTRIES)
def test_every_entry_builds_with_no_arguments(kind, name):
    """The tables promise zero-argument factories; a config must default itself."""
    table, _methods = TABLES[kind]
    assert table[name]() is not None


@pytest.mark.parametrize(("kind", "name"), ENTRIES)
def test_every_entry_provides_its_protocol(kind, name):
    """
    The protocols are not runtime_checkable on purpose - isinstance against one
    only checks the names exist, which is the stringly-typed dispatch these
    tables replace. Check the callables here; mypy checks the shapes, because
    each table is annotated with its protocol at the point it is defined.
    """
    table, methods = TABLES[kind]
    component = table[name]()
    for method in methods:
        assert callable(getattr(component, method)), f"{name} has no {method}()"


def test_every_kind_of_component_is_covered():
    """A fifth pluggable kind should not be able to appear without a table here."""
    assert set(TABLES) == {
        "player model",
        "team model",
        "transfer strategy",
        "squad optimizer",
        "transfer optimizer",
    }


# The command whose --help must list every name in the table. A transfer
# strategy has no flag: which one runs is decided by the move, not by the user,
# so `TRANSFER_STRATEGIES` is deliberately absent from this mapping.
NAMING_FLAGS = {
    "player model": ("predict", "--player-model"),
    "team model": ("predict", "--team-model"),
    "squad optimizer": ("optimize squad", "--squad-optimizer"),
    "transfer optimizer": ("optimize transfers", "--transfer-optimizer"),
}


@pytest.mark.parametrize(
    ("kind", "command", "flag"), [(k, c, f) for k, (c, f) in NAMING_FLAGS.items()]
)
def test_every_name_is_reachable_from_the_command_line(kind, command, flag):
    """
    A table only earns its keep if a name on it reaches an implementation.

    Without this the tables could be read by nothing but this file, which is not
    what CLAUDE.md says they are for.
    """
    table, _methods = TABLES[kind]
    # a narrow terminal wraps a long option name mid-word, so ask for a wide one
    result = CliRunner(env={"COLUMNS": "200"}).invoke(app, [*command.split(), "--help"])
    assert result.exit_code == 0, result.output
    # Rich wraps help text, so compare on whitespace-collapsed output
    help_text = " ".join(result.output.split())
    assert flag in help_text
    for name in table:
        assert name in help_text, f"{flag} does not list {name}"


def test_a_strategy_is_chosen_by_the_move_not_by_a_flag():
    """The one table with no naming flag, recorded so its absence is deliberate."""
    assert "transfer strategy" not in NAMING_FLAGS
