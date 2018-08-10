"""
Class for a player in FPL
"""

from .schema import Player
from .utils import get_player_data, get_predicted_points_for_player


class CandidatePlayer(object):
    """
    player class
    """
    def __init__(self,player):
        """
        initialize either by name or by ID
        """
        data = get_player_data(player)
        for attribute in data.__dir__():
            if not attribute.startswith("_"):
                self.__setattr__(attribute, getattr(data, attribute))
        self.is_starting = True # by default
        self.is_captain = False # by default
        self.is_vice_captain = False # by default


    def calc_expected_points(self, method="EP", gameweek=None):
        """
        get expected points for specified gameweeek
        """
        self.predicted_points = get_predicted_points_for_player(self.player_id)
