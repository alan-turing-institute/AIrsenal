"""
Domain type aliases.

NewType is used only where two values of the same primitive type are routinely
confused and never combined arithmetically. Everything else is a plain alias that
documents intent without forcing wrapper calls at every use.
"""

from typing import NewType, TypeAlias

import numpy as np
import numpy.typing as npt

# Our own player.player_id and the FPL API's player id are different integers for the
# same footballer, and the codebase carries both. This is the distinction most worth
# having the type checker enforce.
PlayerId = NewType("PlayerId", int)
FplPlayerId = NewType("FplPlayerId", int)

# A manager's FPL entry, as opposed to a Premier League club. These collide in
# get_sell_price and in the fetcher, where a wrong one sends real transfers.
FplTeamId = NewType("FplTeamId", int)

FixtureId = NewType("FixtureId", int)

# Deliberately plain aliases. Gameweeks are added and ranged over constantly, so a
# NewType would mean wrapping the result of every `gw + 1` - about 200 sites - to buy
# a distinction nothing actually confuses.
Gameweek: TypeAlias = int
Season: TypeAlias = str  # "2526" for the 2025/26 season
Tag: TypeAlias = str  # groups the rows written by one prediction run
TeamName: TypeAlias = str  # three-letter club code, e.g. "ARS"
Price: TypeAlias = int  # tenths of a million, as the FPL API reports it

FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.int64]
