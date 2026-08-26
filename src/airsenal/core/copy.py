"""Fast deep copies of the plain-data objects used in the optimiser's inner loop."""

from pickle import dumps, loads
from typing import cast


def fastcopy[T](obj: T) -> T:
    """Faster replacement for copy.deepcopy()."""
    return cast("T", loads(dumps(obj, -1)))
