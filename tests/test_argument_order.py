"""
The argument order in CodingConventions.md.

Four groups, in this order: what the argument is about (`player_id`, `player`,
`position`, `team`), which run of the model it reads (`tag`), when (`gameweek`,
`season`), and what it talks to (`fpl_team_id`, `fetcher`, `dbsession`), with
`verbose` last. Anything not named here is unconstrained and conventionally goes
first. Every function in the package follows it; there are no exemptions.

The groups are the point. An earlier version of this order put `tag` between
`gameweek` and `season`, which fitted the signatures marginally better but split
the two time arguments around an unrelated one, and a convention nobody can
recite is not worth the test that enforces it.

`tag` sits above `gameweek` rather than after `season` because it is a required
argument in 83% of the signatures that take it, against 56% for `season`, which
usually carries `= CURRENT_SEASON`. A required argument after a defaulted one is
not expressible in a positional list, so the other placement would have forced
seven more signatures to go keyword-only.

Ten signatures cannot be put in this order by reordering alone, for that same
reason: an optional argument (`position`, `tag`, `gameweek`) ranks above a
required one. Their tails are keyword-only from that point on, which keeps every
default and every requirement as it was.
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
