"""
`game/` imports nothing.

The facts about Fantasy Premier League - what a position is, what a goal is
worth, what "2122" means, what Brentford is abbreviated to - used to live in
`core/` alongside the logger, the console and the query caches, under one
negative rule: "no airsenal-specific dependencies". A negative rule has no
floor, so anything that imported nothing landed there.

A `domain/` package was tried once and merged back (89fc5f2), because it left
`enums.py` behind in `core/` and so "how many points for a goal" was domain
while "what a position is" was core. This test is the difference: the boundary
is a property, not a paragraph. A module here may import the standard library
and its siblings, and nothing else - no pandas, no sqlalchemy, not even
`airsenal.core.logging`. The layers contract in pyproject.toml enforces the
airsenal half by putting `game` at the bottom of the chain; this enforces the
rest.

If something added here needs a logger or a dataframe, that is the signal it is
not a fact about the game.
"""

import ast
import sys
from pathlib import Path

import pytest

GAME = Path(__file__).resolve().parents[2] / "src" / "airsenal" / "game"


def game_modules() -> list[Path]:
    return sorted(GAME.glob("*.py"))


def imported_roots(tree: ast.AST):
    """The top-level package name of every import in the module."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, alias.name.split(".")[0]
        # a relative import can only reach a sibling, which is allowed
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module, node.module.split(".")[0]


@pytest.mark.parametrize("path", game_modules(), ids=lambda p: p.name)
def test_imports_only_the_standard_library_or_a_sibling(path):
    offenders = [
        f"{path.name}: {module}"
        for module, root in imported_roots(ast.parse(path.read_text()))
        if root not in sys.stdlib_module_names
        and not module.startswith("airsenal.game.")
    ]
    assert not offenders, (
        "game/ must import only the standard library and its own modules:\n"
        + "\n".join(offenders)
    )


def test_there_is_something_to_check():
    """A glob that silently matched nothing would make this file pass forever."""
    assert {p.name for p in game_modules()} >= {
        "enums.py",
        "mappings.py",
        "scoring.py",
        "season.py",
    }
