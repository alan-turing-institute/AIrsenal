"""Locating the packaged historical data."""

import os
from importlib.resources import files
from pathlib import Path

FilePath = str | os.PathLike[str]

PACKAGE = "airsenal"
DATA_DIR_NAME = "data"


def data_dir() -> Path:
    """
    The packaged data directory.

    Resolved via the `airsenal` package rather than `airsenal.data`, so the data
    directory does not need an `__init__.py` to make it importable.
    """
    return Path(str(files(PACKAGE))) / DATA_DIR_NAME


def data_file(*parts: str) -> Path:
    """A packaged data file, e.g. `data_file(f"results_{season}.csv")`."""
    return data_dir().joinpath(*parts)


def absences_file(season: str) -> Path:
    """
    The absences CSV for a season.

    Named here rather than in `ingest/absences.py` because `export/absences.py`
    writes the same file.
    """
    return data_file(f"absences_{season}.csv")
