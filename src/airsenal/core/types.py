"""
Domain type aliases.

Plain aliases only. Ids were NewTypes here for a while, but every id in this codebase
arrives from SQLAlchemy or from the FPL API as a plain int, so a NewType would have
meant a cast at each of those boundaries - many hundreds of them - to buy a
distinction that reads fine as a parameter name.
"""

import numpy as np
import numpy.typing as npt

# Gameweeks are added and ranged over constantly, so these stay plain ints and strs.
type Gameweek = int
type Season = str  # "2526" for the 2025/26 season
type Tag = str  # groups the rows written by one prediction run
type TeamName = str  # three-letter club code, e.g. "ARS"
type Price = int  # tenths of a million, as the FPL API reports it

type FloatArray = npt.NDArray[np.float64]
type IntArray = npt.NDArray[np.int64]
