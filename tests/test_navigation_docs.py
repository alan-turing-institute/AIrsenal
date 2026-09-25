"""
The signposts to the code stay in step with it.

`docs/architecture.md` maps every module, `docs/where-to-look.md` maps every
command, and every package says in its docstring what it owns. Each is only
useful while it is complete, so each is checked here.
"""

import ast
import re
from pathlib import Path

import pytest
import typer.main

from airsenal.cli.main import app

ROOT = Path(__file__).parents[1]
PACKAGE = ROOT / "src" / "airsenal"
ARCHITECTURE = ROOT / "docs" / "architecture.md"
WHERE_TO_LOOK = ROOT / "docs" / "where-to-look.md"


def _first_cells(doc: Path, pattern: str) -> set[str]:
    """The backticked first cell of every table row that matches `pattern`."""
    rows = re.findall(rf"^\| `({pattern})` \|", doc.read_text(), flags=re.MULTILINE)
    return set(rows)


def _map_rows() -> set[str]:
    """The files and directories the architecture map has a row for."""
    return _first_cells(ARCHITECTURE, r"[\w/]+(?:\.py|/)")


def _modules() -> list[str]:
    """Every module that is not a package's `__init__.py`, relative to the package."""
    return sorted(
        p.relative_to(PACKAGE).as_posix()
        for p in PACKAGE.rglob("*.py")
        if p.name != "__init__.py"
    )


def _commands(command=None, path=("airsenal",)):
    """Every runnable command, as it is typed."""
    command = command or typer.main.get_command(app)
    if not hasattr(command, "commands"):
        yield " ".join(path)
        return
    for name, sub in command.commands.items():
        yield from _commands(sub, (*path, name))


@pytest.mark.parametrize("module", _modules())
def test_every_module_is_on_the_map(module):
    rows = _map_rows()
    in_a_mapped_directory = any(
        module.startswith(row) for row in rows if row.endswith("/")
    )
    assert module in rows or in_a_mapped_directory, (
        f"{module} has no row in the map in docs/architecture.md"
    )


def test_every_map_row_exists():
    missing = sorted(row for row in _map_rows() if not (PACKAGE / row).exists())
    assert not missing, f"docs/architecture.md maps paths that do not exist: {missing}"


def test_every_command_is_in_where_to_look():
    documented = _first_cells(WHERE_TO_LOOK, r"airsenal[\w -]*")
    commands = set(_commands())
    assert commands - documented == set(), "commands missing from docs/where-to-look.md"
    assert documented - commands == set(), (
        "docs/where-to-look.md lists unknown commands"
    )


@pytest.mark.parametrize(
    "package",
    sorted(
        p.parent.relative_to(PACKAGE).as_posix() for p in PACKAGE.rglob("__init__.py")
    ),
)
def test_every_package_says_what_it_owns(package):
    init = PACKAGE / package / "__init__.py"
    assert ast.get_docstring(ast.parse(init.read_text())), f"{init} has no docstring"
