"""
The argument order in CodingConventions.md.

Four groups, in this order: what the argument is about (`player_id`, `player`,
`position`, `team`), which run of the model it reads (`tag`), when (`gameweek`,
`season`), and what it talks to (`fpl_team_id`, `fetcher`, `dbsession`), with
`verbose` last. Anything not named here is unconstrained and conventionally goes
first. Every function in the package follows it; there are no exemptions.
"""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "airsenal"

# The documented order. A function taking two or more of these must take them in
# this relative order; anything not named here is unconstrained.
ARGUMENT_ORDER = [
    # what it is about
    "player_id",
    "player",
    "position",
    "team",
    # which run
    "tag",
    # when
    "gameweek",
    "season",
    # what it talks to
    "fpl_team_id",
    "fetcher",
    "dbsession",
    "verbose",
]
RANK = {name: index for index, name in enumerate(ARGUMENT_ORDER)}


def source_files():
    return sorted(SRC.rglob("*.py"))


def out_of_order(path):
    """(qualified name, the convention args it takes) for each bad signature."""
    relative = str(path.relative_to(SRC))
    bad = []
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        args = [
            a.arg
            for a in node.args.args + node.args.kwonlyargs
            if a.arg not in ("self", "cls")
        ]
        known = [a for a in args if a in RANK]
        if len(known) < 2:
            continue
        if [RANK[a] for a in known] != sorted(RANK[a] for a in known):
            bad.append((f"{relative}:{node.name}", known))
    return bad


@pytest.mark.parametrize("path", source_files(), ids=lambda p: str(p.relative_to(SRC)))
def test_arguments_are_in_the_documented_order(path):
    """Every function takes its arguments in the documented order."""
    correct = " -> ".join(ARGUMENT_ORDER)
    offenders = [
        f"{name} takes ({', '.join(args)}) - the order is {correct}"
        for name, args in out_of_order(path)
    ]
    assert not offenders, "\n".join(offenders)
