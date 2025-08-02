"""
Class for a player in FPL
"""

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
        self, player, season=CURRENT_SEASON, gameweek=NEXT_GAMEWEEK, dbsession=None
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
        price = pdata.price(season, gameweek)
        if price is None:
            msg = f"Player {self} has no price for season {season}, gameweek {gameweek}"
            raise ValueError(msg)
        self.purchase_price = price
        self.is_starting = True
        self.is_captain = False
        self.is_vice_captain = False
        self.predicted_points = {}
        self.sub_position = None

    def __str__(self):
        return self.name

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
            print(f"No prediction available for {self.name} week {gameweek}")
            return 0.0
        return self.predicted_points[tag][gameweek]
