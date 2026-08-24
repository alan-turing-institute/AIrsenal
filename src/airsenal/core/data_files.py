"""
Locating the packaged historical data.

Around twenty modules resolved these files with
`os.path.dirname(__file__) + "/../data/..."`. That works only while the module
sits exactly one directory below the data, which made moving any of them a
silent failure rather than a loud one: from one level deeper, `../data` is a
path that simply does not exist, so the error talks about a missing file rather
than about the layout. Resolving against the package instead does not care where
the caller lives.

Paths are returned as `Path` rather than `Traversable` because several of these
files are written as well as read (refreshing a season's data is a dev-time job
in a checkout), and because the data is 200 MB of CSV and JSON that was never
going to be loaded from a zipimport anyway.
"""

import os
from importlib.resources import files
from pathlib import Path

# Anything open() accepts - callers pass either a data_file() Path or their own
# string path, and neither should have to convert for the other.
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
