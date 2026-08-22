"""Fast deep copies of the plain-data objects used in the optimiser's inner loop."""

from pickle import dumps, loads
from typing import TypeVar

T = TypeVar("T")


def fastcopy(obj: T) -> T:
    """
    Faster replacement for copy.deepcopy().
    """
    return loads(dumps(obj, -1))
