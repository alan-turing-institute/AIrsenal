"""
Importing a module must not talk to the network or open a database.

Creating an engine, running create_all, or resolving a constant with a query at module
scope all break this, and the symptom is a test suite that cannot be collected without
network access.

These tests are the objective definition of "the package imports cleanly". They run in
a subprocess because an in-process check is defeated by everything conftest has already
imported.
"""

import ast
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "airsenal"

# Modules that fail to import for reasons unrelated to I/O (missing optional
# third-party packages). They are reported, not failed.
OPTIONAL_DEPENDENCY_MODULES = {
    "airsenal.reporting.plots",  # matplotlib, in the "plot" extra
}

_GUARD_SCRIPT = textwrap.dedent(
    '''
    """Import every airsenal module with the network and the database blocked."""
    import importlib
    import json
    import pkgutil
    import socket
    import sys
    import traceback

    GUARD_MARKER = "AIRSENAL_IMPORT_GUARD"


    class ImportGuardError(RuntimeError):
        pass


    def _blocked(what):
        def guard(*args, **kwargs):
            msg = f"{GUARD_MARKER}: {what} at import time"
            raise ImportGuardError(msg)

        return guard


    # socket.socket must stay a class: the stdlib ssl module subclasses it at import.
    # Blocking instantiation rather than replacing the name keeps that working.
    class _BlockedSocket(socket.socket):
        def __init__(self, *args, **kwargs):
            msg = f"{GUARD_MARKER}: network access at import time"
            raise ImportGuardError(msg)


    # Import ssl first so its subclassing happens against the real socket class.
    import ssl  # noqa: F401

    socket.socket = _BlockedSocket
    socket.create_connection = _blocked("network access")

    import sqlalchemy

    sqlalchemy.create_engine = _blocked("database engine creation")

    import airsenal

    violations = {}
    other_failures = {}

    for module in pkgutil.walk_packages(airsenal.__path__, "airsenal."):
        name = module.name
        try:
            importlib.import_module(name)
        except BaseException:
            tb = traceback.format_exc()
            if GUARD_MARKER in tb:
                violations[name] = tb.strip().splitlines()[-1]
            else:
                other_failures[name] = tb.strip().splitlines()[-1]

    print(json.dumps({"violations": violations, "other_failures": other_failures}))
    '''
)


def _run_import_guard():
    result = subprocess.run(
        [sys.executable, "-c", _GUARD_SCRIPT],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(f"import guard subprocess failed:\n{result.stderr}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_importing_airsenal_performs_no_io():
    violations = _run_import_guard()["violations"]
    assert not violations, "modules performing I/O at import time:\n" + "\n".join(
        f"  {name}: {reason}" for name, reason in sorted(violations.items())
    )


def test_no_module_fails_to_import_for_unexpected_reasons():
    """
    Every module in the package imports.

    One that cannot is dead weight. Missing *optional* dependencies are allowed,
    and listed explicitly.
    """
    failures = _run_import_guard()["other_failures"]
    unexpected = {
        name: reason
        for name, reason in failures.items()
        if name not in OPTIONAL_DEPENDENCY_MODULES
    }
    assert not unexpected, "modules that fail to import:\n" + "\n".join(
        f"  {name}: {reason}" for name, reason in sorted(unexpected.items())
    )


def _cached_functions_with_session_params():
    """
    Find functions decorated with lru_cache/cache that take a database session.

    functools caches key on argument identity, so a cache keyed on a Session caches
    on the identity of that object: stale reads after a commit, the session held
    alive forever, and leakage between tests that each have their own session.
    """
    offenders = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            decorators = {
                d.id
                if isinstance(d, ast.Name)
                else d.attr
                if isinstance(d, ast.Attribute)
                else d.func.id
                if isinstance(d, ast.Call) and isinstance(d.func, ast.Name)
                else d.func.attr
                if isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute)
                else ""
                for d in node.decorator_list
            }
            if not decorators & {"lru_cache", "cache"}:
                continue
            args = node.args
            names = {a.arg for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]}
            if names & {"dbsession", "session"}:
                rel = path.relative_to(REPO_ROOT)
                offenders.append(f"{rel}:{node.lineno} {node.name}")
    return offenders


def test_no_cache_is_keyed_on_a_database_session():
    offenders = _cached_functions_with_session_params()
    assert not offenders, "cached functions taking a session:\n" + "\n".join(
        f"  {o}" for o in offenders
    )
