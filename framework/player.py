"""
Class for a player in FPL
"""

from .datastore import DataStore

ds = DataStore()

position_map = { 1: 'keeper',
                 2: 'defender',
                 3: 'midfielder',
                 4: 'forward'
}

class Player(object):
    """
    player class
    """
    def __init__(self,player_id):
        self.player_id = player_id
        self.data = ds.get_current_player_data()[player_id]
        self.name = "{} {}".format(self.data["first_name"],
                                   self.data["second_name"])
        self.cost = self.data['now_cost']
        self.team_data = ds.get_current_team_data()[self.data['team_code']]
        self.team = self.team_data['name']
        self.position = position_map[self.data['element_type']]
