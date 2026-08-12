"""
Optimization using DEAP (Distributed Evolutionary Algorithms in Python) to optimize a
full squad for the start of the season, wildcards, or free hits with a genetic
algorithm.
"""

import random

import numpy as np
from deap import algorithms, base, creator, tools

from airsenal.framework.optimization_utils import (
    DEFAULT_SUB_WEIGHTS,
    get_discounted_squad_score,
)
from airsenal.framework.player import DummyPlayer
from airsenal.framework.schema import Player
from airsenal.framework.squad import TOTAL_PER_POSITION, Squad
from airsenal.framework.utils import (
    CURRENT_SEASON,
    get_predicted_points_for_player,
    list_players,
)


class SquadOpt:
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
    base_squad : Squad, optional
        If given, search for the best squad reachable from this existing squad by
        transferring in at most `max_transfers` new players, rather than building a
        squad from scratch. Players kept from `base_squad` cost nothing (they're
        already owned); players sold contribute their real sale price to the budget
        available for new purchases. By default None (build from scratch).
    bank : int, optional
        Cash in the bank, used together with sale proceeds to fund new purchases when
        `base_squad` is given. Ignored otherwise. By default None.
    max_transfers : int, optional
        Maximum number of players that may differ from `base_squad`. Ignored unless
        `base_squad` is given. By default None.
    root_gw : int, optional
        Gameweek to discount future gameweeks in `gw_range` relative to. Useful when
        `gw_range` represents the remaining weeks of a longer, already-underway
        planning horizon (e.g. a node partway down a multi-week tree search), so
        future points are discounted relative to the true start of that horizon
        rather than to `gw_range[0]`. By default None (falls back to `gw_range[0]`).
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
        base_squad=None,
        bank=None,
        max_transfers=None,
        root_gw=None,
    ):
        self.season = season
        self.gw_range = gw_range
        self.start_gw = min(gw_range)
        self.root_gw = root_gw if root_gw is not None else self.start_gw
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

        self.base_squad = base_squad
        self.bank = bank
        self.max_transfers = max_transfers
        if self.base_squad is not None:
            self._base_ids = {p.player_id for p in self.base_squad.players}
            self._sale_prices = {
                p.player_id: self.base_squad.get_sell_price_for_player(
                    p, gameweek=self.start_gw
                )
                for p in self.base_squad.players
            }

        self.players, self.position_idx = self._get_player_list()
        if remove_zero:
            self._remove_zero_pts()
        self.n_available_players = len(self.players)

        # one entry per genome slot, in the same position-block order _create_individual
        # and _get_mutation_bounds lay genes out in
        self._slot_positions = [
            pos for pos in self.positions for _ in range(self.players_per_position[pos])
        ]
        if self.base_squad is not None:
            self._seed_indices = self._get_seed_indices()

        # Setup DEAP toolbox
        self._setup_deap()

    def _get_seed_indices(self) -> list[int]:
        """One genome index per slot, matching base_squad's own players where
        possible (falling back to a random player in that position if a base squad
        player isn't in the candidate pool, e.g. filtered out by remove_zero). Used
        to seed the initial population close to base_squad - see
        `_create_seeded_individual` for why that's necessary.
        """
        id_to_index = {p.player_id: i for i, p in enumerate(self.players)}
        remaining_by_position: dict[str, list[int]] = {
            pos: [] for pos in self.positions
        }
        for p in self.base_squad.players:
            remaining_by_position[p.position].append(p.player_id)

        seed_indices = []
        for pos in self._slot_positions:
            pos_min, pos_max = self.position_idx[pos]
            idx = None
            while remaining_by_position[pos]:
                pid = remaining_by_position[pos].pop()
                if pid in id_to_index:
                    idx = id_to_index[pid]
                    break
            if idx is None:
                idx = random.randint(pos_min, pos_max)
            seed_indices.append(idx)
        return seed_indices

    def _setup_deap(self):
        """Setup DEAP genetic algorithm components."""
        # Create fitness and individual classes
        # We want to maximize fitness, so weights=(1.0,)
        # Guard against re-registering these with DEAP's process-global creator
        # registry, which happens whenever more than one SquadOpt is constructed in
        # the same process (e.g. once per transfer decision in a tree search) and
        # otherwise emits a RuntimeWarning on every subsequent construction.
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)

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
        if self.base_squad is not None:
            return self._create_seeded_individual()

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

    def _create_seeded_individual(self):
        """Individual seeded from base_squad (see _get_seed_indices), with a random
        number (0 to max_transfers) of slots swapped for a different player in the
        same position slot.

        Without this, individuals are generated the same way as the from-scratch
        case (_create_individual) - entirely at random per position slot. Against a
        real candidate pool that's very unlikely to happen to match most of a
        15-player squad by chance, so with max_transfers small relative to squad
        size, the whole population could end up with zero individuals satisfying
        the transfer cap at all, leaving the GA with no fitness gradient to search
        with.
        """
        individual = list(self._seed_indices)
        n_swaps = random.randint(0, self.max_transfers)
        for slot in random.sample(range(len(individual)), n_swaps):
            pos = self._slot_positions[slot]
            pos_min, pos_max = self.position_idx[pos]
            individual[slot] = random.randint(pos_min, pos_max)
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

    def _build_squad(self, candidate_ids: list) -> Squad | None:
        """Build the Squad represented by a list of player IDs, applying the
        budget/transfer-cap/position rules for this optimization. Returns None if
        any constraint is violated. Used both to score individuals during the GA
        search and to decode the final chosen individual (see `decode_individual`),
        so the two can never disagree about whether a given squad is valid/what it
        costs - they're the same code, not two hand-duplicated copies of it.
        """
        if len(set(candidate_ids)) != len(candidate_ids):
            return None  # same player picked more than once

        if self.base_squad is not None:
            new_in = [pid for pid in candidate_ids if pid not in self._base_ids]
            sold_out = self._base_ids - set(candidate_ids)
            if len(new_in) > self.max_transfers:
                return None  # too many transfers
            budget = self.bank + sum(self._sale_prices[pid] for pid in sold_out)
            squad = Squad(budget=budget, season=self.season)
            for pid in candidate_ids:
                # kept players cost nothing - they're already owned, not re-bought
                price = 0 if pid in self._base_ids else None
                if not squad.add_player(pid, price=price, gameweek=self.start_gw):
                    return None
        else:
            # Make squad from player IDs
            squad = Squad(budget=self.budget, season=self.season)

            # Add selected players to squad
            for pid in candidate_ids:
                if not squad.add_player(pid, gameweek=self.start_gw):
                    return None

        # Fill empty slots with dummy players (if chosen not to optimise full squad)
        for pos in self.positions:
            if self.dummy_per_position[pos] > 0:
                for _ in range(self.dummy_per_position[pos]):
                    dp = DummyPlayer(
                        self.gw_range, self.tag, pos, purchase_price=self.dummy_sub_cost
                    )
                    if not squad.add_player(dp):
                        return None

        if not squad.is_complete():
            return None
        return squad

    def _evaluate_individual(self, individual: list[int]) -> tuple[float]:
        """Evaluate the fitness of an individual (squad)."""
        candidate_ids = [self.players[int(idx)].player_id for idx in individual]
        squad = self._build_squad(candidate_ids)
        if squad is None:
            return (0.0,)

        # Calculate expected points for all gameweeks
        score = get_discounted_squad_score(
            squad,
            self.gw_range,
            self.tag,
            self.root_gw,
            self.bench_boost_gw,
            self.triple_captain_gw,
            sub_weights=self.sub_weights,
        )

        return (score,)

    def decode_individual(self, individual: list[int]) -> Squad:
        """Build the Squad represented by a DEAP individual, e.g. the best_individual
        returned by `optimize()`. Raises RuntimeError if it's not valid - shouldn't
        happen for anything optimize() reports as the best individual, since this
        applies the exact same construction used to score it during the search.
        """
        candidate_ids = [self.players[int(idx)].player_id for idx in individual]
        squad = self._build_squad(candidate_ids)
        if squad is None:
            msg = "Individual does not represent a valid squad"
            raise RuntimeError(msg)
        return squad

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
        players: list[Player] = []
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
        crossover_indpb: float = 0.5,
        mutation_indpb: float = 0.1,
        tournament_size: int = 3,
        verbose: bool = True,
        random_state: int | None = None,
    ) -> tuple[list[int], float]:
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
        crossover_indpb : float
            Independent probability for each attribute to be exchanged in crossover
        mutation_indpb : float
            Independent probability for each attribute to be mutated
        tournament_size : int
            Size of tournament for tournament selection
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

        # Register genetic operators with configurable parameters
        self.toolbox.register("mate", tools.cxUniform, indpb=crossover_indpb)
        self.toolbox.register(
            "mutate",
            tools.mutUniformInt,
            low=self.low_bounds,
            up=self.up_bounds,
            indpb=mutation_indpb,
        )
        self.toolbox.register("select", tools.selTournament, tournsize=tournament_size)

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
        population, _logbook = algorithms.eaSimple(
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


def make_new_squad(
    gw_range,
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
    population_size=100,
    generations=100,
    crossover_prob=0.7,
    mutation_prob=0.3,
    crossover_indpb=0.5,
    mutation_indpb=0.1,
    tournament_size=3,
    random_state=None,
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
    crossover_indpb : float, optional
        Independent probability for each attribute to be exchanged in crossover,
        by default 0.5
    mutation_indpb : float, optional
        Independent probability for each attribute to be mutated, by default 0.1
    tournament_size : int, optional
        Size of tournament for tournament selection, by default 3
    random_state : int, optional
        Random seed for reproducibility, by default None

    Returns
    -------
    airsenal.framework.squad.Squad
        The optimized squad
    """
    # Build optimization problem
    opt_squad = SquadOpt(
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
        crossover_indpb=crossover_indpb,
        mutation_indpb=mutation_indpb,
        tournament_size=tournament_size,
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
                player,
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
                    purchase_price=opt_squad.dummy_sub_cost,
                )
                squad.add_player(dp)
                if verbose:
                    print(dp.position, dp.name, dp.purchase_price / 10)

    if verbose:
        print(f"£{squad.budget / 10}m in the bank")

    return squad
