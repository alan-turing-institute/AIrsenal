"""
The class for an FPL team.
Contains a set of players.
Is able to check that it obeys all constraints.
"""
from operator import itemgetter

from .player import CandidatePlayer

# how many players do we need to add
TOTAL_PER_POSITION = {"GK": 2,
                      "DEF": 5,
                      "MID": 5,
                      "FWD": 3
}

# min/max active players per position

ACTIVE_PER_POSITION = {"GK" : (1,1),
                       "DEF": (3,5),
                       "MID": (3,5),
                       "FWD": (1,3)
}

FORMATIONS = [(3,4,3),(3,5,2),(4,3,3),(4,4,2),(4,5,1),(5,4,1),(5,3,2)]


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
            "GK": 0,
            "DEF": 0,
            "MID": 0,
            "FWD": 0
        }
        self.free_subs = 0
        self.subs_this_week = 0

    def __repr__(self):
        """
        Display the team
        """
        print("\n=== starting 11 ===\n")
        for position in ["GK","DEF","MID","FWD"]:
            print("\n== {} ==\n".format(position))
            for p in self.players:
                if p.position == position and p.is_starting:
                    player_line = "{} ({})".format(p.name, p.team)
                    if p.is_captain:
                        player_line += "(C)"
                    elif p.is_vice_captain:
                        player_line += "(VC)"
                    print(player_line)
        print("\n=== subs ===\n")
        for p in self.players:
            if not p.is_starting:
                    print("{} ({})".format(p.name, p.team))
        return ""

    def is_complete(self):
        """
        See if we have 15 players.
        """
        num_players  = sum(self.num_position.values())
        return num_players == 15

    def add_player(self,p):
        """
        add a player.  Can do it by name or by player_id
        """
        player = CandidatePlayer(p)
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
        self.num_position[player.position] += 1
        self.budget -= player.current_price
        return True


    def remove_player(self, player_id):
        """
        remove player from our list
        """
        for p in self.players:
            if p.player_id == player_id:
                self.budget += p.current_price
                self.num_position[p.position] -= 1
                self.players.remove(p)
                return True
        return False


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
        can_afford = player.current_price <= self.budget
        return can_afford


    def _calc_expected_points(self, method="EP", gameweek=None):
        """
        estimate the expected points for the specified gameweek.
        If no gameweek is specified, it will be the next fixture
        """
        for p in self.players:
            p.calc_expected_points(method, gameweek)
        pass


    def optimize_subs(self,gameweek=None):
        """
        based on pre-calculated expected points,
        choose the best starting 11, obeying constraints.
        """
        # first order all the players by expected points
        player_dict = {"GK": [], "DEF": [], "MID": [], "FWD": []}
        for p in self.players:
            player_dict[p.position].append((p,p.predicted_points))
        for v in player_dict.values():
            v.sort(key=itemgetter(1),reverse=True)
    #    print(player_dict)

        # always sub the second-placed keeper
        player_dict["GK"][1][0].is_starting=False
        best_score = 0.
        best_formation = None
        for f in FORMATIONS:
            self.apply_formation(player_dict, f)
            score = self.total_points_for_starting_11()
            if score > best_score:
                best_score = score
                best_formation = f
        print("Best formation is {}".format(best_formation))
        self.apply_formation(player_dict, best_formation)
        return best_score

    def apply_formation(self, player_dict,formation):
        """
        set players' is_starting to True or False
        depending on specified formation in format e.g.
        (4,4,2)
        """
        for i,pos in enumerate(["DEF","MID","FWD"]):
            for index,player in enumerate(player_dict[pos]):
                if index < formation[i]:
                    player[0].is_starting=True
                else:
                    player[0].is_starting=False


    def total_points_for_starting_11(self):
        """
        simple sum over starting players
        """
        total = 0.
        for player in self.players:
            if player.is_starting:
                total += player.predicted_points
                if player.is_captain:
                    total += player.predicted_points
        return total


    def get_expected_points(self, method="EP", gameweek=None):
        """
        expected points for the starting 11.
        """
        if not self.is_complete():
            raise RuntimeError("Team is incomplete")
        self._calc_expected_points(method,gameweek)
        self.optimize_subs(gameweek)
        self.pick_captains()
        total_score = self.total_points_for_starting_11()
        return total_score

    def pick_captains(self):
        """
        pick the highest two expected points for captain and vice-captain
        """
        player_list = []
        for p in self.players:
            player_list.append((p,p.predicted_points))

        player_list.sort(key=itemgetter(1),reverse=True)
        player_list[0][0].is_captain=True
        player_list[1][0].is_vice_captain=True
