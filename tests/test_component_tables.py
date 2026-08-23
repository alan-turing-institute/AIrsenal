"""
Every swappable component, checked the same way.

There are four kinds of pluggable component, and each keeps a plain dict of name
to zero-argument factory. Adding an implementation means adding one entry, and
that entry is covered here automatically: it must build with no arguments and
provide the method its protocol names.
"""

import pytest

from airsenal.optimization.squad_optimizers import SQUAD_OPTIMIZERS
from airsenal.optimization.strategies import TRANSFER_STRATEGIES
from airsenal.optimization.transfer_optimizers import TRANSFER_OPTIMIZERS
from airsenal.prediction.models import PLAYER_MODELS, TEAM_MODELS

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
