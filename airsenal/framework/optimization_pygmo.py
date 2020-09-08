import pygmo as pg

import numpy as np
import pandas as pd
import uuid

from .utils import (
    NEXT_GAMEWEEK,
    CURRENT_SEASON,
    list_players,
    get_latest_prediction_tag,
    get_predicted_points_for_player,
)
from .team import Team, TOTAL_PER_POSITION


class DummyPlayer:
    """
    To fill squads with placeholders (if not optimising full squad)
    """

    def __init__(self, gw_range, tag, position, price=45, team="XYZ", pts=0):
        self.name = "DUMMY"
        self.position = position
        self.purchase_price = price
        self.team = str(uuid.uuid4())
        self.pts = pts
        self.predicted_points = {tag: {gw: self.pts for gw in gw_range}}
        self.player_id = str(uuid.uuid4())  # dummy id
        self.is_starting = False
        self.is_captain = False
        self.is_vice_captain = False
        self.sub_position = None

    def calc_predicted_points(self, method):
        """
        get expected points from the db.
        Will be a dict of dicts, keyed by method and gameweeek
        """
        pass

    def get_predicted_points(self, gameweek, method):
        """
        get points for a specific gameweek
        """
        return self.pts


# PyGMO User Defined Problem
class OptTeam:
    def __init__(
        self,
        gw_range,
        tag,
        budget=1000,
        dummy_sub_cost=45,
        season=CURRENT_SEASON,
        bench_boost_gw=None,
        triple_captain_gw=None,
        remove_zero=True,  # don't consider players with predicted pts of zero
        players_per_position=TOTAL_PER_POSITION,
    ):
        self.season = season
        self.gw_range = gw_range
        self.start_gw = min(gw_range)
        self.bench_boost_gw = bench_boost_gw
        self.triple_captain_gw = triple_captain_gw

        self.tag = tag
        self.positions = ["GK", "DEF", "MID", "FWD"]
        self.players_per_position = players_per_position
        self.n_opt_players = sum(self.players_per_position.values())
        self.dummy_per_position = (
            self._get_dummy_per_position()
        )  # no. players each position that won't be optimised (just filled with dummies)
        self.dummy_sub_cost = dummy_sub_cost
        self.budget = budget

        self.players, self.position_idx = self._get_player_list()
        if remove_zero:
            self._remove_zero_pts()
        self.n_available_players = len(self.players)

    def fitness(self, player_ids):
        """
        PyGMO required function.
        The objective function to minimise. And constraints to evaluate.
        """
        # Make team from player IDs
        team = Team(budget=self.budget)
        for idx in player_ids:
            team.add_player(
                self.players[int(idx)].player_id,
                season=self.season,
                gameweek=self.start_gw,
            )

        # fill empty slots with dummy players (if chosen not to optimise full squad)
        for pos in self.positions:
            if self.dummy_per_position[pos] > 0:
                for i in range(self.dummy_per_position[pos]):
                    dp = DummyPlayer(
                        self.gw_range, self.tag, pos, price=self.dummy_sub_cost
                    )
                    team.add_player(dp)

        # Check team is valid
        if not team.is_complete():
            return [0]

        #  Calc expected points for all gameweeks
        score = 0.0
        for gw in self.gw_range:
            if gw == self.bench_boost_gw:
                score += team.get_expected_points(gw, self.tag, bench_boost=True)
            elif gw == self.triple_captain_gw:
                score += team.get_expected_points(gw, self.tag, triple_captain=True)
            else:
                score += team.get_expected_points(gw, self.tag)

        return [-score]

    def get_bounds(self):
        """
        PyGMO required function.
        Defines min and max value for each parameter.
        """
        # use previously calculated position index ranges to set the bounds
        # to force all attempted solutions to contain the correct number of
        # players for each position.
        low_bounds = []
        high_bounds = []
        for pos in self.positions:
            low_bounds += [self.position_idx[pos][0]] * self.players_per_position[pos]
            high_bounds += [self.position_idx[pos][1]] * self.players_per_position[pos]

        return (low_bounds, high_bounds)

    def get_nec(self):
        """PyGMO function.
        Defines number of equality constraints."""
        return 0

    def get_nix(self):
        """
        PyGMO function.
        Number of integer dimensions.
        """
        return self.n_opt_players

    def gradient(self, x):
        return pg.estimate_gradient_h(lambda x: self.fitness(x), x)

    def _get_player_list(self):
        """
        Get list of active players at the start of the gameweek range,
        and the id range of players for each position.
        """
        players = []
        change_idx = [0]
        # build players list by position (e.g. all GK, then all DEF etc.)
        for pos in self.positions:
            players += list_players(
                position=pos, season=self.season, gameweek=self.start_gw
            )
            change_idx.append(len(players))

        # min and max idx of players for each position
        position_idx = {
            self.positions[i - 1]: (change_idx[i - 1], change_idx[i] - 1)
            for i in range(1, len(change_idx))
        }
        return players, position_idx

    def _remove_zero_pts(self):
        players = []
        change_idx = [0]
        last_pos = self.positions[0]
        for p in self.players:
            gw_pts = get_predicted_points_for_player(p, self.tag, season=self.season)
            total_pts = sum([pts for gw, pts in gw_pts.items() if gw in self.gw_range])
            if total_pts > 0:
                if p.position(self.season) != last_pos:
                    change_idx.append(len(players))
                    last_pos = p.position(self.season)
                players.append(p)
        change_idx.append(len(players))

        position_idx = {
            self.positions[i - 1]: (change_idx[i - 1], change_idx[i] - 1)
            for i in range(1, len(change_idx))
        }

        self.players = players
        self.position_idx = position_idx

    def _get_dummy_per_position(self):
        dummy_per_position = {}
        for pos in self.positions:
            dummy_per_position[pos] = (
                TOTAL_PER_POSITION[pos] - self.players_per_position[pos]
            )
        return dummy_per_position


def make_new_team(
    gw_range, tag,
    budget=1000,
    players_per_position=TOTAL_PER_POSITION,
    season=CURRENT_SEASON,
    verbose=1,
    bench_boost_gw=None,
    triple_captain_gw=None,
    uda=pg.sga(gen=100),
    population_size=100,
):
    # Build problem
    opt_team = OptTeam(
        gw_range,
        tag,
        budget=budget,
        players_per_position=players_per_position,
        dummy_sub_cost=45,
        season=season,
        bench_boost_gw=bench_boost_gw,
        triple_captain_gw=triple_captain_gw,
        remove_zero=True,  # don't consider players with predicted pts of zero
    )

    prob = pg.problem(opt_team)
    print(prob)

    # Create algorithm to solve problem with
    algo = pg.algorithm(uda=uda)
    algo.set_verbosity(verbose)
    print(algo)

    # population of problems
    pop = pg.population(prob=prob, size=population_size)

    # solve problem
    pop = algo.evolve(pop)        
    print("Best score:", -pop.champion_f[0], "pts")

    # construct optimal team
    team = Team(budget=opt_team.budget)
    for idx in pop.champion_x:
        print(
            opt_team.players[int(idx)].position(CURRENT_SEASON),
            opt_team.players[int(idx)].name,
            opt_team.players[int(idx)].team(CURRENT_SEASON, 1),
            opt_team.players[int(idx)].price(CURRENT_SEASON, 1)/10,
        )
        team.add_player(
            opt_team.players[int(idx)].player_id,
            season=opt_team.season,
            gameweek=opt_team.start_gw,
        )

    # fill empty slots with dummy players (if chosen not to optimise full squad)
    for pos in opt_team.positions:
        if opt_team.dummy_per_position[pos] > 0:
            for i in range(opt_team.dummy_per_position[pos]):
                dp = DummyPlayer(
                    opt_team.gw_range, opt_team.tag, pos, price=opt_team.dummy_sub_cost
                )
                team.add_player(dp)
                print(dp.position, dp.name, dp.purchase_price/10)

    print(f"£{team.budget/10}m in the bank")

    print("=" * 10)
    pts = team.get_expected_points(opt_team.gw_range[0], opt_team.tag)
    print(f"GW{opt_team.gw_range[0]}: {pts:.0f} pts")
    print(team)

    return team
