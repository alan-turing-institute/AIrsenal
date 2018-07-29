"""
The class for an FPL team.
Contains a set of players.
Is able to check that it obeys all constraints.
"""

from .player import Player

# how many players do we need to add
TOTAL_PER_POSITION = {"keeper": 2,
                      "defender": 5,
                      "midfielder": 5,
                      "forward": 3
}

# min/max active players per position

ACTIVE_PER_POSITION = {"keeper" : (1,1),
                       "defender": (3,5),
                       "midfielder": (3,5),
                       "forward": (1,3)
}


class Team(object):
    """
    Team class.
    """
    def __init__(self):
        """
        constructor - start with an initial empty player list,
        and £100M
        """
        self.players = []
        self.budget = 1000
        self.num_position = {
            "keeper": 0,
            "defender": 0,
            "midfielder": 0,
            "forward": 0
        }
        self.free_subs = 0
        self.subs_this_week = 0

    def is_complete(self):
        """
        See if we have 15 players.
        """
        num_players  = sum(self.num_position.values())
        return num_players == 15

    def add_player(self,player_id):
        """
        add a player.
        """
        player = Player(player_id)
        # check if constraints are met
        if not self.check_no_duplicate_player(player):
            print("Already have {} in team".format(player.name))
            return False
        if not self.check_num_in_position(player):
            print("Unable to add player {} - too many {}"\
                  .format(player.name, player.position))
            return False
        if not self.check_cost(player):
            print("Cannot afford player {}".format(player.name))
            return False
        if not self.check_num_per_team(player):
            print("Cannot add {} - too many players from {}"\
                  .format(player.name, player.team))
            return False
        self.players.append(player)
        return True


    def check_no_duplicate_player(self,player):
        """
        Check we don't already have the player.
        """
        for p in self.players:
            if p.player_id == player.player_id:
                return False
        return True


    def check_num_in_position(self,player):
        """
        check we have fewer than the limit of
        num players in the chosen players position.
        """
        return self.num_position[player.position] < \
           TOTAL_PER_POSITION[player.position]


    def check_num_per_team(self, player):
        """
        check we have fewer than 3 players from the same
        team as the specified player.
        """
        num_same_team = 0
        for p in self.players:
            if p.team == player.team:
                num_same_team += 1
                if num_same_team == 3:
                    return False
        return True


    def check_cost(self,player):
        """
        check we can afford the player.
        """
        return player.cost <= self.budget


    def _calc_expected_points(self, gameweek, method="EP"):
        """
        estimate the expected points for the specified gameweek.
        """
        pass


    def optimize_subs(self,gameweek):
        """
        based on pre-calculated expected points,
        choose the best starting 11, obeying constraints.
        """
        # first order all the players by expected points

        pass


    def get_expected_points(self, gameweek, method="EP"):
        """
        expected points for the starting 11.
        """
        if not self.is_complete():
            raise RuntimeError("Team is incomplete")
        self._calc_expected_points(gameweek, method)
        total = 0.
        for player in self.players:
            if player.is_starting():
                total += player.expected_points
        return total
