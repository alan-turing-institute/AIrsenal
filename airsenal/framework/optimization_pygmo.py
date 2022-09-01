"""
Alternative approach to algorithm in optimization_squad using pygmo to create an
optimal squad for the start of the season or for wildcards and free hits.

Usually gives better results than optimization_squad.make_new_squad if run with a
population size of at least 100 and for at least 100 generations.
"""

try:
    import pygmo as pg
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional dependency pygmo not installed. If using conda run "
        "'conda install pygmo'. If not see https://esa.github.io/pygmo2/install.html"
    ) from e


import uuid

from airsenal.framework.optimization_utils import (
    DEFAULT_SUB_WEIGHTS,
    get_discounted_squad_score,
)
from airsenal.framework.squad import TOTAL_PER_POSITION, Squad
from airsenal.framework.utils import (
    CURRENT_SEASON,
    get_predicted_points_for_player,
    list_players,
)


class DummyPlayer:
    """To fill squads with placeholders (if not optimising full squad)."""

    def __init__(self, gw_range, tag, position, price=45, pts=0):
        self.name = "DUMMY"
        self.position = position
        self.purchase_price = price
        # set team to random string so we don't violate max players per team constraint
        self.team = str(uuid.uuid4())
        self.pts = pts
        self.predicted_points = {tag: {gw: self.pts for gw in gw_range}}
        self.player_id = str(uuid.uuid4())  # dummy id
        self.is_starting = False
        self.is_captain = False
        self.is_vice_captain = False
        self.sub_position = None

    def calc_predicted_points(self, tag):
        """
        Needed for compatibility with Squad/other Player classes
        """
        pass

    def get_predicted_points(self, gameweek, tag):
        """
        Get points for a specific gameweek -
        """
        return self.pts


class SquadOpt:
    """Pygmo user defined problem class for optimising a full squad

    Parameters
    ----------
    gw_range : list
        Gameweeks to optimize squad for
    tag : str
        Points prediction tag to use
    budget : int, optional
        Total budget for squad times 10,  by default 1000
    players_per_position : dict
        No. of players to optimize in each position, by default
        airsenal.framework.squad.TOTAL_PER_POSITION
    season : str
        Season to optimize for, by default airsenal.framework.utils.CURRENT_SEASON
    bench_boost_gw : int
        Gameweek to play benfh boost, by default None
    triple_captain_gw : int
        Gameweek to play triple captaiin, by default None,
    remove_zero : bool
        If True don't consider players with predicted pts of zero, by default True
    sub_weights : dict
        Weighting to give to substitutes in optimization, by default
        {"GK": 0.01, "Outfield": (0.4, 0.1, 0.02)},
    dummy_sub_cost : int, optional
        If not optimizing a full squad the price of each player that is not being
        optimized. For example, if you are optimizing 12 out of 15 players, the
        effective budget for optimizinig the squad will be
        budget - (15 -12) * dummy_sub_cost, by default 45
    """

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
        sub_weights=DEFAULT_SUB_WEIGHTS,
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
        # no. players each position that won't be optimised (just filled with dummies)
        self.dummy_per_position = self._get_dummy_per_position()
        self.dummy_sub_cost = dummy_sub_cost
        self.budget = budget
        self.sub_weights = sub_weights

        self.players, self.position_idx = self._get_player_list()
        if remove_zero:
            self._remove_zero_pts()
        self.n_available_players = len(self.players)

    def fitness(self, player_ids):
        """PyGMO required function. The objective function to minimise.
        In this case:
            - 0 if the proposed squad isn't valid
            - weghted sum of gameweek points otherwise
        """
        # Make squad from player IDs
        squad = Squad(budget=self.budget, season=self.season)
        for idx in player_ids:
            add_ok = squad.add_player(
                self.players[int(idx)].player_id,
                gameweek=self.start_gw,
            )
            if not add_ok:
                return [0]

        # fill empty slots with dummy players (if chosen not to optimise full squad)
        for pos in self.positions:
            if self.dummy_per_position[pos] > 0:
                for _ in range(self.dummy_per_position[pos]):
                    dp = DummyPlayer(
                        self.gw_range, self.tag, pos, price=self.dummy_sub_cost
                    )
                    add_ok = squad.add_player(dp)
                    if not add_ok:
                        return [0]

        # Check squad is valid, if not return fitness of zero
        if not squad.is_complete():
            return [0]

        #  Calc expected points for all gameweeks
        score = get_discounted_squad_score(
            squad,
            self.gw_range,
            self.tag,
            self.gw_range[0],
            self.bench_boost_gw,
            self.triple_captain_gw,
            sub_weights=self.sub_weights,
        )

        return [-score]

    def get_bounds(self):
        """PyGMO required function. Defines min and max value for each parameter."""
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
        """PyGMO function. Defines number of equality constraints."""
        return 0

    def get_nix(self):
        """PyGMO function. Number of integer dimensions."""
        return self.n_opt_players

    def gradient(self, x):
        """PyGMO function - estimate gradient"""
        return pg.estimate_gradient_h(lambda x: self.fitness(x), x)

    def _get_player_list(self):
        """Get list of active players at the start of the gameweek range,
        and the id range of players for each position.
        """
        players = []
        change_idx = [0]
        # build players list by position (i.e. all GK, then all DEF etc.)
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
        """Exclude players with zero predicted points."""
        players = []
        # change_idx stores the indices of where the player positions change in the new
        # player list
        change_idx = [0]
        last_pos = self.positions[0]
        for p in self.players:
            gw_pts = get_predicted_points_for_player(p, self.tag, season=self.season)
            total_pts = sum(pts for gw, pts in gw_pts.items() if gw in self.gw_range)
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
        """No. of dummy players per position needed to complete the squad (if not
        optimising the full squad)
        """
        return {
            pos: (TOTAL_PER_POSITION[pos] - self.players_per_position[pos])
            for pos in self.positions
        }


def make_new_squad_pygmo(
    gw_range,
    tag,
    budget=1000,
    players_per_position=TOTAL_PER_POSITION,
    season=CURRENT_SEASON,
    verbose=1,
    bench_boost_gw=None,
    triple_captain_gw=None,
    remove_zero=True,  # don't consider players with predicted pts of zero
    sub_weights={"GK": 0.01, "Outfield": (0.4, 0.1, 0.02)},
    dummy_sub_cost=45,
    uda=pg.sga(gen=100),
    population_size=100,
    **kwargs,
):
    """Optimize a full initial squad using any PyGMO-compatible algorithm.

    Parameters
    ----------
    gw_range : list
        Gameweeks to optimize squad for
    tag : str
        Points prediction tag to use
    budget : int, optional
        Total budget for squad times 10,  by default 1000
    players_per_position : dict
        No. of players to optimize in each position, by default
        airsenal.framework.squad.TOTAL_PER_POSITION
    season : str
        Season to optimize for, by default airsenal.framework.utils.CURRENT_SEASON
    verbose : int
        Verbosity of optimization algorithm, by default 1
    bench_boost_gw : int
        Gameweek to play benfh boost, by default None
    triple_captain_gw : int
        Gameweek to play triple captaiin, by default None,
    remove_zero : bool
        If True don't consider players with predicted pts of zero, by default True
    sub_weights : dict
        Weighting to give to substitutes in optimization, by default
        {"GK": 0.01, "Outfield": (0.4, 0.1, 0.02)},
    dummy_sub_cost : int, optional
        If not optimizing a full squad the price of each player that is not being
        optimized. For example, if you are optimizing 12 out of 15 players, the
        effective budget for optimizinig the squad will be
        budget - (15 -12) * dummy_sub_cost, by default 45
    uda : class, optional
        PyGMO compatible algorithm class, by default pg.sga(gen=100)
    population_size : int, optional
        Number of candidate solutions in each generation of the optimization,
        by default 100

    Returns
    -------
    airsenal.framework.squad.Squad
        The optimized squad
    """
    # Build problem
    opt_squad = SquadOpt(
        gw_range,
        tag,
        budget=budget,
        players_per_position=players_per_position,
        dummy_sub_cost=dummy_sub_cost,
        season=season,
        bench_boost_gw=bench_boost_gw,
        triple_captain_gw=triple_captain_gw,
        remove_zero=remove_zero,  # don't consider players with predicted pts of zero
        sub_weights=sub_weights,
    )
    prob = pg.problem(opt_squad)

    # Create algorithm to solve problem with
    algo = pg.algorithm(uda=uda)
    algo.set_verbosity(verbose)

    # population of problems
    pop = pg.population(prob=prob, size=population_size)

    # solve problem
    pop = algo.evolve(pop)
    if verbose > 0:
        print("Best score:", -pop.champion_f[0], "pts")

    # construct optimal squad
    squad = Squad(budget=opt_squad.budget, season=season)
    for idx in pop.champion_x:
        if verbose > 0:
            print(
                opt_squad.players[int(idx)].position(season),
                opt_squad.players[int(idx)].name,
                opt_squad.players[int(idx)].team(season, 1),
                opt_squad.players[int(idx)].price(season, 1) / 10,
            )
        squad.add_player(
            opt_squad.players[int(idx)].player_id,
            gameweek=opt_squad.start_gw,
        )

    # fill empty slots with dummy players (if chosen not to optimise full squad)
    for pos in opt_squad.positions:
        if opt_squad.dummy_per_position[pos] > 0:
            for _ in range(opt_squad.dummy_per_position[pos]):
                dp = DummyPlayer(
                    opt_squad.gw_range,
                    opt_squad.tag,
                    pos,
                    price=opt_squad.dummy_sub_cost,
                )
                squad.add_player(dp)
                if verbose > 0:
                    print(dp.position, dp.name, dp.purchase_price / 10)
    if verbose > 0:
        print(f"£{squad.budget/10}m in the bank")

    return squad
