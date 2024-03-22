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


class CandidatePlayer(object):
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
            pdata = get_player(player, self.dbsession)
        self.player_id = pdata.player_id
        self.name = pdata.name
        self.season = season
        self.team = pdata.team(season, gameweek)
        self.position = pdata.position(season)
        self.purchase_price = pdata.price(season, gameweek)
        self.is_starting = True  # by default
        self.is_captain = False  # by default
        self.is_vice_captain = False  # by default
        self.predicted_points = {}
        self.sub_position = None

    def __str__(self):
        return self.name

    def calc_predicted_points(self, tag):
        """
        get expected points from the db.
        Will be a dict of dicts, keyed by tag and gameweeek
        """
        if tag not in self.predicted_points.keys():
            self.predicted_points[tag] = get_predicted_points_for_player(
                self.player_id, tag, season=self.season, dbsession=self.dbsession
            )

    def get_predicted_points(self, gameweek, tag):
        """
        get points for a specific gameweek
        """
        if tag not in self.predicted_points.keys():
            self.calc_predicted_points(tag)
        if gameweek not in self.predicted_points[tag].keys():
            print(f"No prediction available for {self.name} week {gameweek}")
            return 0.0
        return self.predicted_points[tag][gameweek]
