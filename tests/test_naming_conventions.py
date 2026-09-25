"""
The naming conventions in CodingConventions.md, enforced.

A gameweek is `gameweek`, a list of them is `gameweeks`, a count of them is
`n_gameweeks`, and a position or chip is written as its enum. None of those
fail at runtime if broken, so they are asserted here instead.
"""

import ast
import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "airsenal"

# Position is a StrEnum, so a bare literal works and neither the tests nor the type
# checker notice one. These two modules are the boundary and keep their literals:
# game/enums.py defines them, and game/mappings.py maps the FPL API's own integers
# and abbreviations - including "MID" for Middlesbrough, which is a club, not a
# midfielder.
POSITION_LITERALS = {"GK", "DEF", "MID", "FWD"}
POSITION_LITERAL_EXEMPT = {"game/enums.py", "game/mappings.py"}

# Chip is a StrEnum too. game/enums.py defines the values. The rest are not chip
# references at all: game/mappings.py and
# apply/transfers.py hold the FPL API's own spellings - the ones it puts in
# `active_chip` and the fields its transfers endpoint takes - and the strings in
# export/db_dump.py are Transaction column names being written to a CSV header.
CHIP_LITERALS = {"wildcard", "free_hit", "bench_boost", "triple_captain"}
CHIP_LITERAL_EXEMPT = {
    "game/enums.py",
    "game/mappings.py",
    "apply/transfers.py",
    "export/db_dump.py",
}

# banned name -> what to use instead
BANNED_PARAMETERS = {
    "gw_range": "gameweeks",
    "gameweek_range": "gameweeks",
    "weeks_ahead": "n_gameweeks",
    "num_gameweeks": "n_gameweeks",
    "gw_ahead": "n_gameweeks",
    "num_weeks": "n_gameweeks",
    "pred_tag": "prediction_tag",
    "num_match_to_use": "n_matches_to_use",
    "n_games_to_use": "n_matches_to_use",
    "apifetcher": "fetcher",
}


def squashed(name):
    """A name with the separators taken out, so spellings of it compare equal."""
    return name.lower().replace("_", "").replace("-", "")


# The same names in every other spelling: `weeks_ahead`, `WeeksAhead` and
# `--weeks-ahead` are all banned.
BANNED_SPELLINGS = {squashed(old): new for old, new in BANNED_PARAMETERS.items()}


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


# A gameweek is spelled out, in every name and in every part of one: `gameweek`,
# `gameweeks`, `n_gameweeks`, and where a qualifier is needed `gameweek_start` or
# `bench_boost_gameweek`. `week` counts as an abbreviation too: nothing in the
# package measures a calendar week.
GAMEWEEK_ABBREVIATIONS = {"gw", "gws", "week", "weeks"}


def name_parts(name):
    """The words in a name, whether it is snake_case, CamelCase or a --flag."""
    parts = []
    for chunk in re.split(r"[-_]", name.lstrip("-")):
        parts += re.findall(r"[A-Z]+(?![a-z])|[A-Z][a-z]*|[a-z]+|\d+", chunk)
    return [part.lower() for part in parts]


def abbreviates_gameweek(name):
    return any(part in GAMEWEEK_ABBREVIATIONS for part in name_parts(name))


def named_nodes(tree):
    """Every name a module introduces, and the CLI flags it spells out."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            yield node.name, node.lineno
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            yield node.id, node.lineno
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
            yield node.attr, node.lineno
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.startswith("--")
        ):
            yield node.value, node.lineno


@pytest.mark.parametrize("path", source_files(), ids=lambda p: str(p.relative_to(SRC)))
def test_no_banned_names_in_any_spelling(path):
    """A banned name stays banned as a type alias, a property or a flag."""
    offenders = [
        f"{path.relative_to(SRC)}:{lineno} {name} "
        f"- use {BANNED_SPELLINGS[squashed(name)]}"
        for name, lineno in named_nodes(ast.parse(path.read_text()))
        if squashed(name) in BANNED_SPELLINGS
    ]
    assert not offenders, "\n".join(offenders)


@pytest.mark.parametrize("path", source_files(), ids=lambda p: str(p.relative_to(SRC)))
def test_a_gameweek_is_never_abbreviated(path):
    """
    Nothing is named `gw` or `week`, in any part of any name.

    The checks above and `tests/test_argument_order.py` match whole names, so
    this is what catches a name such as `next_gw`.
    """
    tree = ast.parse(path.read_text())
    named = list(named_nodes(tree))
    named += [
        (arg.arg, arg.lineno)
        for node in functions_in(path)
        for arg in parameters_of(node)
    ]
    offenders = [
        f"{path.relative_to(SRC)}:{lineno} {name} - a gameweek is spelled out"
        for name, lineno in named
        if abbreviates_gameweek(name)
    ]
    assert not offenders, "\n".join(offenders)


@pytest.mark.parametrize("path", source_files(), ids=lambda p: str(p.relative_to(SRC)))
def test_gameweek_is_never_an_int_or_list_union(path):
    """A parameter called `gameweek` is one gameweek, never `int | list[int]`."""
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
def test_a_chip_is_written_as_the_enum(path):
    """`Chip.WILDCARD`, not `"wildcard"`."""
    relative = str(path.relative_to(SRC))
    if relative in CHIP_LITERAL_EXEMPT:
        pytest.skip(f"{relative} is the string boundary")
    offenders = [
        f'{relative}:{node.lineno} "{node.value}" - use Chip.{node.value.upper()}'
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.Constant) and node.value in CHIP_LITERALS
    ]
    assert not offenders, "\n".join(offenders)


@pytest.mark.parametrize("path", source_files(), ids=lambda p: str(p.relative_to(SRC)))
def test_a_position_is_written_as_the_enum(path):
    """`Position.GK`, not `"GK"`."""
    relative = str(path.relative_to(SRC))
    if relative in POSITION_LITERAL_EXEMPT:
        pytest.skip(f"{relative} is the string boundary")
    offenders = [
        f'{relative}:{node.lineno} "{node.value}" - use Position.{node.value}'
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.Constant) and node.value in POSITION_LITERALS
    ]
    assert not offenders, "\n".join(offenders)
