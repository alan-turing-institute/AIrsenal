"""
Optimization using DEAP (Distributed Evolutionary Algorithms in Python) to optimize a
full squad for the start of the season, wildcards, or free hits with a genetic
algorithm.
"""

import random
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
from deap import algorithms, base, creator, tools
from sqlalchemy.orm import Session

from airsenal.core.enums import Position
from airsenal.core.logging import get_logger
from airsenal.core.season import CURRENT_SEASON
from airsenal.db.queries.players import list_players
from airsenal.db.queries.predictions import get_predicted_points_for_player
from airsenal.optimization.config import DEFAULT_SUB_WEIGHTS, GeneticAlgorithmConfig
from airsenal.optimization.squad_score import get_discounted_squad_score
from airsenal.squad.player import DummyPlayer
from airsenal.squad.squad import TOTAL_PER_POSITION, Squad

if TYPE_CHECKING:
    from airsenal.db.models import Player

logger = get_logger(__name__)


def _ensure_deap_types() -> None:
    """
    Register the DEAP fitness and individual classes, once per process.

    creator.create writes into module-level state, so calling it per SquadOpt
    instance made DEAP warn about overwriting an existing class on every
    instantiation, and left the result dependent on which test ran first. The names
    are prefixed to avoid colliding with any other DEAP user in the process.
    """
    if not hasattr(creator, "AirsenalFitnessMax"):
        creator.create("AirsenalFitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "AirsenalIndividual"):
        creator.create("AirsenalIndividual", list, fitness=creator.AirsenalFitnessMax)


class SquadOpt:
    """DEAP-based optimization class for optimising a fantasy football squad

    Parameters
    ----------
    gameweeks : list
        Gameweeks to optimize squad for
    tag : str
        Points prediction tag to use
    budget : int, optional
        Total budget for squad times 10,  by default 1000
    players_per_position : dict
        No. of players to optimize in each position, by default
        airsenal.squad.squad.TOTAL_PER_POSITION
    season : str
        Season to optimize for, by default airsenal.core.season.CURRENT_SEASON
    bench_boost_gw : int
        Gameweek to play bench boost, by default None
    triple_captain_gw : int
        Gameweek to play triple captain, by default None,
    remove_zero : bool
        If True don't consider players with predicted pts of zero, by default True
    sub_weights : dict
        Weighting to give to substitutes in optimization, by default
        SubWeights() - see airsenal.optimization.config.
    dummy_sub_cost : int, optional
        If not optimizing a full squad the price of each player that is not being
        optimized. For example, if you are optimizing 12 out of 15 players, the
        effective budget for optimizing the squad will be
        budget - (15 -12) * dummy_sub_cost, by default 45
    """

    def __init__(
        self,
        gameweeks,
        tag,
        budget=1000,
        dummy_sub_cost=45,
        season=CURRENT_SEASON,
        bench_boost_gw=None,
        triple_captain_gw=None,
        remove_zero=True,  # don't consider players with predicted pts of zero
        players_per_position=TOTAL_PER_POSITION,
        sub_weights=DEFAULT_SUB_WEIGHTS,
        dbsession: Session | None = None,
    ):
        # Held on the optimiser, never on a Squad: a Squad crosses the
        # multiprocessing queue and a Session cannot be pickled.
        self.dbsession = dbsession
        self.season = season
        self.gameweeks = gameweeks
        self.start_gw = min(gameweeks)
        self.bench_boost_gw = bench_boost_gw
        self.triple_captain_gw = triple_captain_gw

        self.tag = tag
        self.positions = list(Position.back_to_front())
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
        _ensure_deap_types()

        self.toolbox = base.Toolbox()

        # Register functions for creating individuals and population
        self.toolbox.register("individual", self._create_individual)
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )

        # Register evaluation function
        self.toolbox.register("evaluate", self._evaluate_individual)

        # Store mutation bounds for later use in optimize method
        self.low_bounds, self.up_bounds = self._get_mutation_bounds()

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

        return creator.AirsenalIndividual(individual)

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

    def _evaluate_individual(self, individual: list[int]) -> tuple[float]:
        """Evaluate the fitness of an individual (squad)."""
        # Make squad from player IDs
        squad = Squad(budget=self.budget, season=self.season)

        # Add selected players to squad
        for idx in individual:
            add_ok = squad.add_player(
                self.players[int(idx)].player_id,
                gameweek=self.start_gw,
                dbsession=self.dbsession,
            )
            if not add_ok:
                return (0.0,)  # Invalid squad

        # Fill empty slots with dummy players (if chosen not to optimise full squad)
        for pos in self.positions:
            if self.dummy_per_position[pos] > 0:
                for _ in range(self.dummy_per_position[pos]):
                    dp = DummyPlayer(
                        self.gameweeks,
                        self.tag,
                        pos,
                        purchase_price=self.dummy_sub_cost,
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
            self.gameweeks,
            self.tag,
            self.gameweeks[0],
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
                position=pos,
                season=self.season,
                gameweek=self.start_gw,
                dbsession=self.dbsession,
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
        players: list[Player] = []
        # change_idx stores the indices of where the player positions change in the new
        # player list
        change_idx = [0]
        last_pos = self.positions[0]
        for p in self.players:
            gw_pts = get_predicted_points_for_player(
                p, self.tag, season=self.season, dbsession=self.dbsession
            )
            total_pts = sum(pts for gw, pts in gw_pts.items() if gw in self.gameweeks)
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
        self, config: GeneticAlgorithmConfig | None = None
    ) -> tuple[list[int], float]:
        """
        Run the genetic algorithm.

        Parameters
        ----------
        config : GeneticAlgorithmConfig, optional
            Population size, generations, operator probabilities and seed. Defaults
            to GeneticAlgorithmConfig(); see airsenal.optimization.config.

        Returns
        -------
        tuple[list[int], float]
            The best individual found and its fitness.
        """
        config = config if config is not None else GeneticAlgorithmConfig()
        if config.random_state is not None:
            random.seed(config.random_state)
            np.random.seed(config.random_state)

        # Register genetic operators with configurable parameters
        self.toolbox.register("mate", tools.cxUniform, indpb=config.crossover_indpb)
        self.toolbox.register(
            "mutate",
            tools.mutUniformInt,
            low=self.low_bounds,
            up=self.up_bounds,
            indpb=config.mutation_indpb,
        )
        self.toolbox.register(
            "select", tools.selTournament, tournsize=config.tournament_size
        )

        # Create initial population
        population = self.toolbox.population(n=config.population_size)

        # Statistics tracking
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # Hall of fame to track best individuals
        hall_of_fame = tools.HallOfFame(1)

        # Run the genetic algorithm
        population, _logbook = algorithms.eaSimple(
            population,
            self.toolbox,
            cxpb=config.crossover_prob,
            mutpb=config.mutation_prob,
            ngen=config.generations,
            stats=stats,
            halloffame=hall_of_fame,
            verbose=config.verbose,
        )

        # Return best individual and its fitness
        best_individual = hall_of_fame[0]
        best_fitness = best_individual.fitness.values[0]

        return best_individual, best_fitness


def make_new_squad(
    gameweeks,
    tag,
    budget=1000,
    players_per_position=TOTAL_PER_POSITION,
    season=CURRENT_SEASON,
    verbose=True,
    bench_boost_gw=None,
    triple_captain_gw=None,
    remove_zero=True,  # don't consider players with predicted pts of zero
    sub_weights=DEFAULT_SUB_WEIGHTS,
    dummy_sub_cost=45,
    ga_config: GeneticAlgorithmConfig | None = None,
    dbsession: Session | None = None,
):
    """Optimize a full initial squad using DEAP genetic algorithm.

    Parameters
    ----------
    gameweeks : list
        Gameweeks to optimize squad for
    tag : str
        Points prediction tag to use
    budget : int, optional
        Total budget for squad times 10,  by default 1000
    players_per_position : dict
        No. of players to optimize in each position, by default
        airsenal.squad.squad.TOTAL_PER_POSITION
    season : str
        Season to optimize for, by default airsenal.core.season.CURRENT_SEASON
    verbose : bool
        Whether the underlying DEAP genetic algorithm should print its own
        per-generation progress to stdout, by default True
    bench_boost_gw : int
        Gameweek to play bench boost, by default None
    triple_captain_gw : int
        Gameweek to play triple captain, by default None,
    remove_zero : bool
        If True don't consider players with predicted pts of zero, by default True
    sub_weights : dict
        Weighting to give to substitutes in optimization, by default
        SubWeights() - see airsenal.optimization.config.
    dummy_sub_cost : int, optional
        If not optimizing a full squad the price of each player that is not being
        optimized. For example, if you are optimizing 12 out of 15 players, the
        effective budget for optimizing the squad will be
        budget - (15 -12) * dummy_sub_cost, by default 45
    ga_config : GeneticAlgorithmConfig, optional
        Genetic algorithm settings; see airsenal.optimization.config.

    Returns
    -------
    airsenal.squad.squad.Squad
        The optimized squad
    """
    # Build optimization problem
    opt_squad = SquadOpt(
        gameweeks,
        tag,
        budget=budget,
        players_per_position=players_per_position,
        dummy_sub_cost=dummy_sub_cost,
        season=season,
        bench_boost_gw=bench_boost_gw,
        triple_captain_gw=triple_captain_gw,
        remove_zero=remove_zero,
        sub_weights=sub_weights,
        dbsession=dbsession,
    )

    # Run optimization
    ga_config = ga_config if ga_config is not None else GeneticAlgorithmConfig()
    if verbose != ga_config.verbose:
        ga_config = replace(ga_config, verbose=verbose)
    best_individual, best_fitness = opt_squad.optimize(ga_config)

    logger.debug("Best score: %s pts", best_fitness)

    # Construct optimal squad
    squad = Squad(budget=opt_squad.budget, season=season)
    for idx in best_individual:
        player = opt_squad.players[int(idx)]
        logger.debug(
            "%s %s %s %s",
            player.position(season),
            player,
            player.team(season, 1),
            player.price(season, 1) / 10,
        )
        squad.add_player(
            opt_squad.players[int(idx)].player_id,
            gameweek=opt_squad.start_gw,
            dbsession=dbsession,
        )

    # Fill empty slots with dummy players (if chosen not to optimise full squad)
    for pos in opt_squad.positions:
        if opt_squad.dummy_per_position[pos] > 0:
            for _ in range(opt_squad.dummy_per_position[pos]):
                dp = DummyPlayer(
                    opt_squad.gameweeks,
                    opt_squad.tag,
                    pos,
                    purchase_price=opt_squad.dummy_sub_cost,
                )
                squad.add_player(dp)
                logger.debug("%s %s %s", dp.position, dp.name, dp.purchase_price / 10)

    logger.debug("£%sm in the bank", squad.budget / 10)

    return squad
