"""
The naming conventions in CodingConventions.md, enforced.

A gameweek is `gameweek`, a list of them is `gameweeks`, a count of them is
`n_gameweeks`, and a position or chip is written as its enum. None of those
fail at runtime if broken, so they are asserted here instead.
"""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "airsenal"

# Position is a StrEnum, so a bare literal still works and nothing fails when one
# is written instead of the enum - which is why this test exists rather than the
# type checker catching it. These two modules are the boundary and keep their
# literals: game/enums.py defines them, and
# game/mappings.py maps the FPL API's own integers and abbreviations - including
# "MID" for Middlesbrough, which is a club, not a midfielder.
POSITION_LITERALS = {"GK", "DEF", "MID", "FWD"}
POSITION_LITERAL_EXEMPT = {"game/enums.py", "game/mappings.py"}

# old name -> what to use instead
BANNED_PARAMETERS = {
    "gw_range": "gameweeks",
    "gameweek_range": "gameweeks",
    "weeks_ahead": "n_gameweeks",
    "num_gameweeks": "n_gameweeks",
    "pred_tag": "prediction_tag",
    "num_match_to_use": "n_matches_to_use",
    "n_games_to_use": "n_matches_to_use",
    "apifetcher": "fetcher",
}


def source_files():
    return sorted(SRC.rglob("*.py"))


def functions_in(path: Path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            yield node


def parameters_of(node):
    args = node.args
    for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs]:
        yield arg
    for arg in (args.vararg, args.kwarg):
        if arg is not None:
            yield arg


@pytest.mark.parametrize("path", source_files(), ids=lambda p: str(p.relative_to(SRC)))
def test_no_banned_parameter_names(path):
    offenders = [
        f"{path.relative_to(SRC)}:{arg.lineno} {node.name}({arg.arg}=...) "
        f"- use {BANNED_PARAMETERS[arg.arg]}"
        for node in functions_in(path)
        for arg in parameters_of(node)
        if arg.arg in BANNED_PARAMETERS
    ]
    assert not offenders, "\n".join(offenders)


@pytest.mark.parametrize("path", source_files(), ids=lambda p: str(p.relative_to(SRC)))
def test_gameweek_is_never_an_int_or_list_union(path):
    """
    A parameter called `gameweek` must be one gameweek.

    Three functions took `gameweek: int | list[int]` and branched on the type,
    which meant every caller had to be read to know which it was passing.
    """
    offenders = []
    for node in functions_in(path):
        for arg in parameters_of(node):
            if arg.arg != "gameweek" or arg.annotation is None:
                continue
            annotation = ast.unparse(arg.annotation)
            if "list" in annotation and "int" in annotation:
                offenders.append(
                    f"{path.relative_to(SRC)}:{arg.lineno} "
                    f"{node.name}(gameweek: {annotation}) - split into two parameters"
                )
    assert not offenders, "\n".join(offenders)


@pytest.mark.parametrize("path", source_files(), ids=lambda p: str(p.relative_to(SRC)))
def test_a_position_is_written_as_the_enum(path):
    """
    `Position.GK`, not `"GK"`.

    A bare literal is not wrong today - Position subclasses str - so nothing
    fails when one is written. That is what this test is for.
    """
    relative = str(path.relative_to(SRC))
    if relative in POSITION_LITERAL_EXEMPT:
        pytest.skip(f"{relative} is the string boundary")
    offenders = [
        f'{relative}:{node.lineno} "{node.value}" - use Position.{node.value}'
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.Constant) and node.value in POSITION_LITERALS
    ]
    assert not offenders, "\n".join(offenders)
