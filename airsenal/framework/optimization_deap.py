"""
Alternative approach to algorithm in optimization_squad using DEAP to create an
optimal squad for the start of the season or for wildcards and free hits.

DEAP (Distributed Evolutionary Algorithms in Python) is a pure Python package
that provides genetic algorithm functionality without requiring conda installation.

This implementation uses DEAP's built-in genetic operators:
- Crossover: tools.cxUniform (uniform crossover)
- Mutation: tools.mutUniformInt (uniform integer mutation with position-specific bounds)
- Selection: tools.selTournament (tournament selection)

OPTIMIZED PARAMETER SETS:
- FAST: population_size=30, generations=50, cx=0.8, mut=0.2, tournament=2
- BALANCED: population_size=80, generations=80, cx=0.75, mut=0.25, tournament=4
- HIGH_QUALITY: population_size=150, generations=150, cx=0.8, mut=0.2, tournament=5
- HIGH_EXPLORATION: population_size=50, generations=50, cx=0.6, mut=0.4, tournament=2
"""

import random
import uuid
from typing import List, Optional, Tuple

import numpy as np

try:
    from deap import algorithms, base, creator, tools
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional dependency deap not installed. Run 'pip install deap' to install."
    ) from e

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


class SquadOptDEAP:
    """DEAP-based optimization class for optimising a fantasy football squad

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
        Gameweek to play bench boost, by default None
    triple_captain_gw : int
        Gameweek to play triple captain, by default None,
    remove_zero : bool
        If True don't consider players with predicted pts of zero, by default True
    sub_weights : dict
        Weighting to give to substitutes in optimization, by default
        {"GK": 0.01, "Outfield": (0.4, 0.1, 0.02)},
    dummy_sub_cost : int, optional
        If not optimizing a full squad the price of each player that is not being
        optimized. For example, if you are optimizing 12 out of 15 players, the
        effective budget for optimizing the squad will be
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

        # Setup DEAP toolbox
        self._setup_deap()

    def _setup_deap(self):
        """Setup DEAP genetic algorithm components."""
        # Create fitness and individual classes
        # We want to maximize fitness, so weights=(1.0,)
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()

        # Register functions for creating individuals and population
        self.toolbox.register("individual", self._create_individual)
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )

        # Create bounds for mutation - each position has its own bounds
        low_bounds, up_bounds = self._get_mutation_bounds()

        # Register genetic operators using DEAP built-ins
        self.toolbox.register("evaluate", self._evaluate_individual)
        self.toolbox.register("mate", tools.cxUniform, indpb=0.5)
        self.toolbox.register(
            "mutate", tools.mutUniformInt, low=low_bounds, up=up_bounds, indpb=0.1
        )
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def _create_individual(self):
        """Create a valid individual (chromosome) representing a squad selection."""
        individual = []

        # For each position, select the required number of players
        for pos in self.positions:
            pos_min, pos_max = self.position_idx[pos]
            n_players = self.players_per_position[pos]

            # Randomly select players for this position
            selected_players = random.sample(
                range(pos_min, pos_max + 1), min(n_players, pos_max - pos_min + 1)
            )
            individual.extend(selected_players)

        return creator.Individual(individual)

    def _get_mutation_bounds(self):
        """Get lower and upper bounds for each gene for mutation."""
        low_bounds = []
        up_bounds = []

        # For each position, add bounds for each player slot
        for pos in self.positions:
            pos_min, pos_max = self.position_idx[pos]
            n_players = self.players_per_position[pos]

            # Add bounds for each player in this position
            low_bounds.extend([pos_min] * n_players)
            up_bounds.extend([pos_max] * n_players)

        return low_bounds, up_bounds

    def _evaluate_individual(self, individual: List[int]) -> Tuple[float]:
        """Evaluate the fitness of an individual (squad)."""
        # Make squad from player IDs
        squad = Squad(budget=self.budget, season=self.season)

        # Add selected players to squad
        for idx in individual:
            add_ok = squad.add_player(
                self.players[int(idx)].player_id,
                gameweek=self.start_gw,
            )
            if not add_ok:
                return (0.0,)  # Invalid squad

        # Fill empty slots with dummy players (if chosen not to optimise full squad)
        for pos in self.positions:
            if self.dummy_per_position[pos] > 0:
                for _ in range(self.dummy_per_position[pos]):
                    dp = DummyPlayer(
                        self.gw_range, self.tag, pos, price=self.dummy_sub_cost
                    )
                    add_ok = squad.add_player(dp)
                    if not add_ok:
                        return (0.0,)  # Invalid squad

        # Check squad is valid, if not return fitness of zero
        if not squad.is_complete():
            return (0.0,)

        # Calculate expected points for all gameweeks
        score = get_discounted_squad_score(
            squad,
            self.gw_range,
            self.tag,
            self.gw_range[0],
            self.bench_boost_gw,
            self.triple_captain_gw,
            sub_weights=self.sub_weights,
        )

        return (score,)

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

    def optimize(
        self,
        population_size: int = 100,
        generations: int = 100,
        crossover_prob: float = 0.7,
        mutation_prob: float = 0.3,
        verbose: bool = True,
        random_state: Optional[int] = None,
    ) -> Tuple[List[int], float]:
        """Run the genetic algorithm optimization.

        Parameters
        ----------
        population_size : int
            Size of the population
        generations : int
            Number of generations to run
        crossover_prob : float
            Probability of crossover
        mutation_prob : float
            Probability of mutation
        verbose : bool
            Whether to print progress
        random_state : int, optional
            Random seed for reproducibility

        Returns
        -------
        Tuple[List[int], float]
            Best individual (player indices) and its fitness score
        """
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)

        # Create initial population
        population = self.toolbox.population(n=population_size)

        # Statistics tracking
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # Hall of fame to track best individuals
        hall_of_fame = tools.HallOfFame(1)

        # Run the genetic algorithm
        population, logbook = algorithms.eaSimple(
            population,
            self.toolbox,
            cxpb=crossover_prob,
            mutpb=mutation_prob,
            ngen=generations,
            stats=stats,
            halloffame=hall_of_fame,
            verbose=verbose,
        )

        # Return best individual and its fitness
        best_individual = hall_of_fame[0]
        best_fitness = best_individual.fitness.values[0]

        return best_individual, best_fitness


def make_new_squad_deap(
    gw_range,
    tag,
    budget=1000,
    players_per_position=TOTAL_PER_POSITION,
    season=CURRENT_SEASON,
    verbose=True,
    bench_boost_gw=None,
    triple_captain_gw=None,
    remove_zero=True,  # don't consider players with predicted pts of zero
    sub_weights={"GK": 0.01, "Outfield": (0.4, 0.1, 0.02)},
    dummy_sub_cost=45,
    population_size=100,
    generations=100,
    crossover_prob=0.7,
    mutation_prob=0.3,
    random_state=None,
    **kwargs,
):
    """Optimize a full initial squad using DEAP genetic algorithm.

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
    verbose : bool
        Whether to print optimization progress, by default True
    bench_boost_gw : int
        Gameweek to play bench boost, by default None
    triple_captain_gw : int
        Gameweek to play triple captain, by default None,
    remove_zero : bool
        If True don't consider players with predicted pts of zero, by default True
    sub_weights : dict
        Weighting to give to substitutes in optimization, by default
        {"GK": 0.01, "Outfield": (0.4, 0.1, 0.02)},
    dummy_sub_cost : int, optional
        If not optimizing a full squad the price of each player that is not being
        optimized. For example, if you are optimizing 12 out of 15 players, the
        effective budget for optimizing the squad will be
        budget - (15 -12) * dummy_sub_cost, by default 45
    population_size : int, optional
        Number of candidate solutions in each generation of the optimization,
        by default 100
    generations : int, optional
        Number of generations to run the genetic algorithm, by default 100
    crossover_prob : float, optional
        Probability of crossover between individuals, by default 0.7
    mutation_prob : float, optional
        Probability of mutation for each individual, by default 0.3
    random_state : int, optional
        Random seed for reproducibility, by default None

    Returns
    -------
    airsenal.framework.squad.Squad
        The optimized squad
    """
    # Build optimization problem
    opt_squad = SquadOptDEAP(
        gw_range,
        tag,
        budget=budget,
        players_per_position=players_per_position,
        dummy_sub_cost=dummy_sub_cost,
        season=season,
        bench_boost_gw=bench_boost_gw,
        triple_captain_gw=triple_captain_gw,
        remove_zero=remove_zero,
        sub_weights=sub_weights,
    )

    # Run optimization
    best_individual, best_fitness = opt_squad.optimize(
        population_size=population_size,
        generations=generations,
        crossover_prob=crossover_prob,
        mutation_prob=mutation_prob,
        verbose=verbose,
        random_state=random_state,
    )

    if verbose:
        print(f"Best score: {best_fitness} pts")

    # Construct optimal squad
    squad = Squad(budget=opt_squad.budget, season=season)
    for idx in best_individual:
        if verbose:
            player = opt_squad.players[int(idx)]
            print(
                player.position(season),
                player.name,
                player.team(season, 1),
                player.price(season, 1) / 10,
            )
        squad.add_player(
            opt_squad.players[int(idx)].player_id,
            gameweek=opt_squad.start_gw,
        )

    # Fill empty slots with dummy players (if chosen not to optimise full squad)
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
                if verbose:
                    print(dp.position, dp.name, dp.purchase_price / 10)

    if verbose:
        print(f"£{squad.budget / 10}m in the bank")

    return squad
