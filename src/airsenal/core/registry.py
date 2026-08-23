"""
Looking an implementation up by name.

Model and algorithm choices reach the code as strings from the command line. Each
kind of component keeps a plain dict of name to zero-argument factory, typed
against the protocol it satisfies, and looks entries up with `lookup` so that an
unknown name is an error listing the valid ones.
"""

from collections.abc import Mapping


class ConfigError(ValueError):
    """
    An unusable name or option came from the command line.

    Distinct from a plain ValueError so the CLI can report it as a bad option
    rather than as a crash, without also swallowing genuine bugs.
    """


def lookup[T](table: Mapping[str, T], name: str, kind: str) -> T:
    """The entry registered under `name`, or a ConfigError naming the valid ones."""
    try:
        return table[name]
    except KeyError:
        msg = f"Unknown {kind} '{name}'. Choose from: {', '.join(sorted(table))}"
        raise ConfigError(msg) from None
