"""
How a player is represented inside a squad.

`CandidatePlayer` is a real player, with a price and predicted points.
`DummyPlayer` fills a slot the search has not decided yet. `SquadPlayer` is
either - it is what a `Squad` holds fifteen of.
"""

import uuid
from collections.abc import Iterable
from typing import Any

from sqlalchemy.orm import Session

from airsenal.core.logging import get_logger
from airsenal.db.models import Player
from airsenal.db.queries.gameweeks import next_gameweek
from airsenal.db.queries.players import get_player
from airsenal.db.queries.predictions import get_predicted_points_for_player
from airsenal.game.season import CURRENT_SEASON

logger = get_logger(__name__)


class CandidatePlayer:
    """
    A real player the optimizer can buy, hold or sell.

    Wraps the database `Player` with what a search needs and the row does not
    have: a purchase price, predicted points per gameweek, and where in the
    lineup it has been placed.
    """

    def __init__(
        self,
        player: Player | str | int,
        season: str = CURRENT_SEASON,
        gameweek: int | None = None,
        purchase_price: int | None = None,
        dbsession: Session | None = None,
    ) -> None:
        """
        Initialize from a `Player`, a name or a player ID.

        Team, position and price are read from the player's attributes for
        `season` and `gameweek`; a player missing any of them is an error rather
        than a partially built candidate.
        """
        # Deliberately NOT resolved to a real session here. CandidatePlayer instances
        # are held by Squad, and Squad is pickled onto the multiprocessing queue by the
        # transfer optimiser (and by fastcopy). A live Session cannot be pickled, and
        # eagerly resolving one here would also open a database connection for every
        # candidate player considered during the search. Callees resolve None
        # themselves, in whichever process ends up needing a session.
        gameweek = next_gameweek() if gameweek is None else gameweek
        self.dbsession = dbsession
        if isinstance(player, Player):
            pdata = player
        else:
            p = get_player(player, self.dbsession)
            if p is None:
                msg = f"Player {player} not found in database"
                raise ValueError(msg)
            pdata = p
        self.player_id = pdata.player_id
        self.name = pdata.name
        self.display_name = pdata.display_name
        self.season = season
        team = pdata.team(season, gameweek)
        if team is None:
            msg = f"Player {self} has no team for season {season}, gameweek {gameweek}"
            raise ValueError(msg)
        self.team = team
        position = pdata.position(season)
        if position is None:
            msg = f"Player {self} has no position for season {season}"
            raise ValueError(msg)
        self.position = position
        if purchase_price is None:
            purchase_price = pdata.price(season, gameweek)
            if purchase_price is None:
                msg = f"{self} has no price for season {season}, gameweek {gameweek}"
                raise ValueError(msg)
        self.purchase_price = purchase_price
        self.is_starting = True
        self.is_captain = False
        self.is_vice_captain = False
        self.predicted_points: dict[str, dict[int, float]] = {}
        self.sub_position: int | None = None

    def __str__(self) -> str:
        return self.display_name or self.name

    def __getstate__(self) -> dict[str, Any]:
        """
        Drop the database session when pickling.

        A Session is bound to a connection and cannot be pickled, but Squad - which
        holds CandidatePlayers - is pickled onto the transfer optimiser's
        multiprocessing queue and by fastcopy. The session is process-local anyway, so
        an unpickled player resolves one in whichever process it wakes up in.
        """
        state = self.__dict__.copy()
        state["dbsession"] = None
        return state

    def calc_predicted_points(self, tag: str) -> None:
        """
        get expected points from the db.
        Will be a dict of dicts, keyed by tag and gameweeek
        """
        if tag not in self.predicted_points:
            self.predicted_points[tag] = get_predicted_points_for_player(
                self.player_id, tag, season=self.season, dbsession=self.dbsession
            )

    def get_predicted_points(self, gameweek: int, tag: str) -> float:
        """
        get points for a specific gameweek
        """
        if tag not in self.predicted_points:
            self.calc_predicted_points(tag)
        if gameweek not in self.predicted_points[tag]:
            logger.warning("No prediction available for %s week %s", self, gameweek)
            return 0.0
        return self.predicted_points[tag][gameweek]


class DummyPlayer:
    """
    To fill squads with placeholders for optimisation (if not optimising full squad).
    """

    def __init__(
        self,
        gameweeks: Iterable[int],
        tag: str,
        position: str,
        purchase_price: int = 45,
        pts: float = 0,
    ) -> None:
        self.name = "DUMMY"
        self.display_name = "DUMMY"
        self.position = position
        self.purchase_price = purchase_price
        # set team to random string so we don't violate max players per team constraint
        self.team = str(uuid.uuid4())
        self.pts = pts
        self.predicted_points: dict[str, dict[int, float]] = {
            tag: dict.fromkeys(gameweeks, self.pts)
        }
        # negative so it can never collide with a real (positive) player id
        self.player_id = -(uuid.uuid4().int % (2**31))
        self.is_starting = False
        self.is_captain = False
        self.is_vice_captain = False
        self.sub_position: int | None = None
        self.season = "DUMMY"

    def calc_predicted_points(self, tag: str) -> None:
        """
        Needed for compatibility with Squad/other Player classes
        """

    def get_predicted_points(self, gameweek: int, tag: str) -> float:  # noqa: ARG002
        """
        Get points for a specific gameweek -
        """
        return self.pts


# Squad holds both: a real player, or a placeholder used when the optimiser is
# not choosing the whole squad.
type SquadPlayer = CandidatePlayer | DummyPlayer


def bench_position(player: SquadPlayer) -> int:
    """
    Where a benched player sits in the substitution order.

    Set by Squad.order_substitutes. Raises rather than returning None, so
    sorting on it before the lineup has been optimized fails here rather than
    inside sorted().
    """
    if player.sub_position is None:
        msg = f"{player} has no bench position - optimize the lineup first"
        raise RuntimeError(msg)
    return player.sub_position
