"""
Every notebook still imports things that exist.

`CodingConventions.md` points a new contributor at `notebooks/` as a place to
start experimenting, and the models are the thing people most want to experiment
with - so a notebook that cannot reach `airsenal.prediction` is a broken on-ramp.

They break silently. Nothing imports a notebook, no test ran one, and ruff does
not lint them, so three package renames went by and six notebooks were left
importing `airsenal.domain.season`, `airsenal.prediction.config` and
`fit_player_data` from a module it had moved out of. This resolves every
`from airsenal... import ...` in every notebook against the installed package,
which is the specific thing that rots.

It deliberately does not run them, lint them or check anything else: they are
exploratory, and the only promise made here is that their imports are real.
"""

import ast
import importlib
import json
from pathlib import Path

import pytest

NOTEBOOKS = sorted((Path(__file__).resolve().parents[1] / "notebooks").glob("*.ipynb"))


def airsenal_imports(path):
    """(module, name) for every `from airsenal... import name` in a notebook."""
    imports = []
    cells = json.loads(path.read_text())["cells"]
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        try:
            tree = ast.parse("".join(cell["source"]))
        except SyntaxError:
            # a half-written cell in a notebook someone is still editing is not
            # what this is looking for
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "airsenal"
            ):
                imports.extend((node.module, alias.name) for alias in node.names)
    return imports


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_imports_resolve(path):
    missing = []
    for module_name, name in airsenal_imports(path):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            missing.append(f"{module_name} does not exist")
            continue
        if not hasattr(module, name):
            missing.append(f"{module_name} has no {name}")
    assert not missing, f"{path.name}:\n" + "\n".join(dict.fromkeys(missing))


def test_there_are_notebooks_to_check():
    """A glob that matches nothing would make every test above vacuously pass."""
    assert NOTEBOOKS
