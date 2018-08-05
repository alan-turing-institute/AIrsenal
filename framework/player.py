"""
Class for a player in FPL
"""

from .schema import Player
from .utils import get_player_data


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
