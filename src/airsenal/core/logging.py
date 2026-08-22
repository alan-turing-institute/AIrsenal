"""Logging setup.

Kept apart from console.py so that code which only needs a logger does not pull
in Rich - which is what let rendering leak into the database layer.
"""

import logging

from rich.logging import RichHandler

from airsenal.core.console import console

_LOGGER_NAME = "airsenal"


def configure_logging(level: int | str = logging.INFO) -> None:
    """Configure the AIrsenal logger to write through Rich.

    Designed for a CLI user reading a terminal, not an operator reading a log
    file: no timestamps, logger names, or file paths - just the message,
    colour-coded by level, with Rich markup in the message rendered.

    Parameters
    ----------
    level : int | str
        Minimum level to display (e.g. ``logging.DEBUG`` or ``"DEBUG"``).
    """
    handler = RichHandler(
        console=console,
        show_time=False,
        show_level=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
    )
    handler.setFormatter(logging.Formatter("%(message)s"))

    logger = logging.getLogger(_LOGGER_NAME)
    logger.handlers = [handler]
    logger.setLevel(level)
    logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Get an AIrsenal logger for the given module, e.g. ``__name__``."""
    return logging.getLogger(name)
