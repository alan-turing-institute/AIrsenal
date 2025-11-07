"""
Class for a player in FPL
"""

import uuid

from airsenal.framework.schema import Player
from airsenal.framework.season import CURRENT_SEASON
from airsenal.framework.utils import (
    NEXT_GAMEWEEK,
    get_player,
    get_predicted_points_for_player,
)


class CandidatePlayer:
    """
    player class
    """

    def __init__(
        self,
        player,
        season=CURRENT_SEASON,
        gameweek=NEXT_GAMEWEEK,
        purchase_price: int | None = None,
        dbsession=None,
    ):
        """
        initialize either by name or by ID
        """
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
        self.sub_position = None

    def __str__(self):
        return self.display_name or self.name

    def calc_predicted_points(self, tag):
        """
        get expected points from the db.
        Will be a dict of dicts, keyed by tag and gameweeek
        """
        if tag not in self.predicted_points:
            self.predicted_points[tag] = get_predicted_points_for_player(
                self.player_id, tag, season=self.season, dbsession=self.dbsession
            )

    def get_predicted_points(self, gameweek, tag):
        """
        get points for a specific gameweek
        """
        if tag not in self.predicted_points:
            self.calc_predicted_points(tag)
        if gameweek not in self.predicted_points[tag]:
            print(f"No prediction available for {self} week {gameweek}")
            return 0.0
        return self.predicted_points[tag][gameweek]


class DummyPlayer:
    """
    To fill squads with placeholders for optimisation (if not optimising full squad).
    """

    def __init__(self, gw_range, tag, position, purchase_price=45, pts=0):
        self.name = "DUMMY"
        self.display_name = "DUMMY"
        self.position = position
        self.purchase_price = purchase_price
        # set team to random string so we don't violate max players per team constraint
        self.team = str(uuid.uuid4())
        self.pts = pts
        self.predicted_points = {tag: dict.fromkeys(gw_range, self.pts)}
        self.player_id = str(uuid.uuid4())  # dummy id
        self.is_starting = False
        self.is_captain = False
        self.is_vice_captain = False
        self.sub_position = None
        self.season = "DUMMY"

    def calc_predicted_points(self, tag):
        """
        Needed for compatibility with Squad/other Player classes
        """

    def get_predicted_points(self, gameweek, tag):  # noqa: ARG002
        """
        Get points for a specific gameweek -
        """
        return self.pts
