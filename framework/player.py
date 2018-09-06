"""
Class for a player in FPL
"""

from .schema import Player
from .utils import get_player_data, get_predicted_points_for_player


class CandidatePlayer(object):
    """
    player class
    """

    def __init__(self, player):
        """
        initialize either by name or by ID
        """
        data = get_player_data(player)
        for attribute in data.__dir__():
            if not attribute.startswith("_"):
                self.__setattr__(attribute, getattr(data, attribute))
        self.is_starting = True  # by default
        self.is_captain = False  # by default
        self.is_vice_captain = False  # by default
        self.predicted_points = {}

    def calc_predicted_points(self, method="AIv1"):
        """
        get expected points from the db.
        Will be a dict of dicts, keyed by method and gameweeek
        """
        if not method in self.predicted_points.keys():
            self.predicted_points[method] = get_predicted_points_for_player(
                self.player_id, method
            )

    def get_predicted_points(self, gameweek, method="AIv1"):
        """
        get points for a specific gameweek
        """
        if not method in self.predicted_points.keys():
            self.calc_predicted_points(method)
        if not gameweek in self.predicted_points[method].keys():
            print(
                "No prediction available for {} week {}".format(
                    self.data.name, gameweek
                )
            )
            return 0.
        return self.predicted_points[method][gameweek]
