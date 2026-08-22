"""Shared parsing for command-line options."""

import typer


def parse_options(values: list[str] | None) -> dict[str, str]:
    """
    Turn repeated `--set-x key=value` options into a dict.

    An unknown key is rejected downstream by the registry, which lists the valid
    ones - hyperparameters used to be silently dropped instead.
    """
    options = {}
    for item in values or []:
        key, sep, value = item.partition("=")
        if not sep:
            msg = f"Expected key=value, got {item!r}"
            raise typer.BadParameter(msg)
        options[key.strip()] = value.strip()
    return options
