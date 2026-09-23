"""At debug level a log line names the module that wrote it."""

import logging
from collections.abc import Iterator

import pytest

from airsenal.core.console import console
from airsenal.core.logging import configure_logging, get_logger


@pytest.fixture(autouse=True)
def _restore_logger() -> Iterator[None]:
    logger = logging.getLogger("airsenal")
    handlers, level = logger.handlers, logger.level
    try:
        yield
    finally:
        logger.handlers, logger.level = handlers, level


def _emit(level: int | str) -> str:
    configure_logging(level)
    with console.capture() as capture:
        get_logger("airsenal.prediction.run").warning("fitting")
    return capture.get()


@pytest.mark.parametrize("level", [logging.DEBUG, "DEBUG"])
def test_debug_lines_name_their_module(level):
    assert "airsenal.prediction.run:" in _emit(level)


def test_other_levels_print_the_bare_message():
    output = _emit(logging.INFO)
    assert "fitting" in output
    assert "airsenal.prediction.run" not in output
