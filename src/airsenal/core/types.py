"""
Domain type aliases.

Plain aliases only. Ids were NewTypes here for a while, but every id in this codebase
arrives from SQLAlchemy or from the FPL API as a plain int, so a NewType would have
meant a cast at each of those boundaries - many hundreds of them - to buy a
distinction that reads fine as a parameter name.
"""

from typing import TypeAlias

import numpy as np
import numpy.typing as npt

# Gameweeks are added and ranged over constantly, so these stay plain ints and strs.
Gameweek: TypeAlias = int
Season: TypeAlias = str  # "2526" for the 2025/26 season
Tag: TypeAlias = str  # groups the rows written by one prediction run
TeamName: TypeAlias = str  # three-letter club code, e.g. "ARS"
Price: TypeAlias = int  # tenths of a million, as the FPL API reports it

FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.int64]
