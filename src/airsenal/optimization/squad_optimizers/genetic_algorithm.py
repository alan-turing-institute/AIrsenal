"""
The DEAP genetic algorithm itself: pick a whole squad, generation by generation.

Inside `squad_optimizers/` the way the tree search lives inside
`transfer_optimizers/`, and the only module in the package that touches DEAP -
which is why it is the one exempted from mypy's `disallow_untyped_calls`.
`genetic.py` is the `SquadOptimizer` wrapper; nothing else should import this.
"""

import random
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
from deap import algorithms, base, creator, tools
from sqlalchemy.orm import Session

from airsenal.core.logging import get_logger
from airsenal.db.models import Player
from airsenal.db.queries.players import list_players
from airsenal.db.queries.predictions import get_predicted_points_for_player
from airsenal.game.enums import Position
from airsenal.game.season import CURRENT_SEASON
from airsenal.optimization.squad_score import (
    SquadScoringConfig,
    get_discounted_squad_score,
)
from airsenal.squad.player import DummyPlayer
from airsenal.squad.squad import TOTAL_PER_POSITION, Squad, SubWeights

logger = get_logger(__name__)


@dataclass(frozen=True)
class GeneticAlgorithmConfig:
    """
    Settings for the search below.

    Beside the algorithm they configure, the way `TreeSearchConfig` sits beside
    the tree search: how the algorithm works is not something the pipeline or the
    CLI has to know about.
    """

    population_size: int = 100
    generations: int = 100
    crossover_prob: float = 0.7
    mutation_prob: float = 0.3
    crossover_indpb: float = 0.5
    mutation_indpb: float = 0.1
    tournament_size: int = 3
    random_state: int | None = None
    verbose: bool = False

    def scaled(self, num_iterations: int) -> "GeneticAlgorithmConfig":
        """
        Population and generations both set from one number.

        How this optimizer reads `SquadRequest.effort`: the transfer search has a
        single --num-iterations knob. Questionable - the two control different
        things - but it is at least explicit here.
        """
        return replace(self, population_size=num_iterations, generations=num_iterations)


# Called after each generation, with the best fitness found so far. A generation
# is the only unit of this search whose count is known before it starts: how many
# individuals one evaluates depends on which of them crossover and mutation
# touched, so only per-generation progress can be sized in advance.
type GenerationReporter = Callable[[float], None]


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
    """
    DEAP-based optimization of a fantasy football squad.

    Args:
        budget: Total squad budget in tenths of a million, so 1000 is £100m.
        players_per_position: How many players to optimize in each position.
            Anything short of a full squad leaves the rest as dummies.
        remove_zero: If True, players with a predicted total of zero points are
            not considered at all.
        sub_weights: How much a substitute's points are worth relative to a
            starter's; see `airsenal.squad.squad`.
        dummy_sub_cost: Price assumed for each player not being optimized, so
            optimizing 12 of 15 leaves `budget - 3 * dummy_sub_cost` to spend.
    """

    def __init__(
        self,
        gameweeks: list[int],
        tag: str,
        budget: int = 1000,
        dummy_sub_cost: int = 45,
        season: str = CURRENT_SEASON,
        bench_boost_gw: int | None = None,
        triple_captain_gw: int | None = None,
        # don't consider players with predicted pts of zero
        remove_zero: bool = True,
        players_per_position: dict[str, int] = TOTAL_PER_POSITION,
        sub_weights: SubWeights | None = None,
        dbsession: Session | None = None,
    ) -> None:
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
        self.sub_weights = (
            sub_weights if sub_weights is not None else SquadScoringConfig().sub_weights
        )

        self.players, self.position_idx = self._get_player_list()
        if remove_zero:
            self._remove_zero_pts()
        self.n_available_players = len(self.players)

        self._setup_deap()

    def _setup_deap(self) -> None:
        _ensure_deap_types()

        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", self._create_individual)
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )
        self.toolbox.register("evaluate", self._evaluate_individual)

        # Needed by the mutation operator, registered in optimize()
        self.low_bounds, self.up_bounds = self._get_mutation_bounds()

    def _create_individual(self) -> list[int]:
        """
        A random starting squad, as indices into `self.players`.

        The list is grouped by position and each group is drawn from that
        position's contiguous slice, so an individual is always positionally
        valid even before the budget is checked.
        """
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

    def _get_mutation_bounds(self) -> tuple[list[int], list[int]]:
        """
        Per-gene index bounds, so a mutation cannot change a player's position.

        Each gene is bounded to the slice of `self.players` for the position that
        slot holds.
        """
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
        """
        The squad's discounted score, or 0.0 if it is not a legal squad.

        Over budget, too many players from one club, or a duplicated player all
        score zero rather than raising, which is how the GA discards them.
        """
        squad = Squad(budget=self.budget, season=self.season)

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
                        pos,
                        self.tag,
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

    def _get_player_list(self) -> tuple[list[Player], dict[Position, tuple[int, int]]]:
        """
        The players active at the start of the window, and where each position sits.

        The list is grouped by position, so the second return value is the
        (first, last) index of each position's block within it.
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

    def _remove_zero_pts(self) -> None:
        """Exclude players with zero predicted points."""
        players: list[Player] = []
        # change_idx stores the indices of where the player positions change in the new
        # player list
        change_idx = [0]
        last_pos: str | None = self.positions[0]
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

    def _get_dummy_per_position(self) -> dict[Position, int]:
        """How many dummies each position needs to bring the squad up to full size."""
        return {
            pos: (TOTAL_PER_POSITION[pos] - self.players_per_position[pos])
            for pos in self.positions
        }

    def optimize(
        self,
        config: GeneticAlgorithmConfig | None = None,
        on_generation: GenerationReporter | None = None,
    ) -> tuple[list[int], float]:
        """
        Run the genetic algorithm.

        Args:
            on_generation: Called after each generation with the best fitness so
                far. Given one, the search is run a generation at a time so that
                it can report; the result is the same either way.

        Returns:
            The best individual found, and its fitness.
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

        population = self.toolbox.population(n=config.population_size)

        # Statistics tracking
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # Hall of fame to track best individuals
        hall_of_fame = tools.HallOfFame(1)

        if on_generation is None:
            algorithms.eaSimple(
                population,
                self.toolbox,
                cxpb=config.crossover_prob,
                mutpb=config.mutation_prob,
                ngen=config.generations,
                stats=stats,
                halloffame=hall_of_fame,
                verbose=config.verbose,
            )
        else:
            self._run_generations(
                population, config, stats, hall_of_fame, on_generation
            )

        best_individual = hall_of_fame[0]
        best_fitness = best_individual.fitness.values[0]

        return best_individual, best_fitness

    def _run_generations(
        self,
        population: list["creator.AirsenalIndividual"],
        config: GeneticAlgorithmConfig,
        stats: tools.Statistics,
        hall_of_fame: tools.HallOfFame,
        on_generation: GenerationReporter,
    ) -> None:
        """
        Run one generation per `eaSimple` call, reporting the best score after each.

        `eaSimple` offers no hook inside its own loop, so reporting per
        generation means calling it one generation at a time rather than owning a
        copy of the loop. Re-entering it with an already-evaluated population
        evaluates nothing, so the extra cost is a hall-of-fame update and a stats
        pass, and the result is identical for a given seed - `test_optimization_squad`
        pins that.

        DEAP's own per-generation printing stays off here: it would repeat the
        logbook header on every call, and a caller that asked to be told about
        each generation is reporting progress itself.
        """
        for _ in range(config.generations):
            # eaSimple replaces the population in place, so each call carries on
            # from where the last one left off
            algorithms.eaSimple(
                population,
                self.toolbox,
                cxpb=config.crossover_prob,
                mutpb=config.mutation_prob,
                ngen=1,
                stats=stats,
                halloffame=hall_of_fame,
                verbose=False,
            )
            on_generation(hall_of_fame[0].fitness.values[0])


def make_new_squad(
    gameweeks: list[int],
    tag: str,
    budget: int = 1000,
    players_per_position: dict[str, int] = TOTAL_PER_POSITION,
    season: str = CURRENT_SEASON,
    bench_boost_gw: int | None = None,
    triple_captain_gw: int | None = None,
    # don't consider players with predicted pts of zero
    remove_zero: bool = True,
    sub_weights: SubWeights | None = None,
    dummy_sub_cost: int = 45,
    ga_config: GeneticAlgorithmConfig | None = None,
    on_generation: GenerationReporter | None = None,
    dbsession: Session | None = None,
) -> Squad:
    """
    Optimize a full initial squad using the DEAP genetic algorithm.

    Everything up to `dummy_sub_cost` is passed straight to `SquadOpt`, which
    documents it. Beyond that:

    Args:
        ga_config: Population size, generations, operator probabilities and seed.
        on_generation: Called after each generation with the best score so far,
            for a caller that wants to show progress. An alternative to the
            config's `verbose`, which prints DEAP's own logbook instead.
    """
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

    ga_config = ga_config if ga_config is not None else GeneticAlgorithmConfig()
    best_individual, best_fitness = opt_squad.optimize(ga_config, on_generation)

    logger.debug("Best score: %s pts", best_fitness)

    # Construct optimal squad
    squad = Squad(budget=opt_squad.budget, season=season)
    for idx in best_individual:
        player = opt_squad.players[int(idx)]
        price = player.price(1, season)
        logger.debug(
            "%s %s %s %s",
            player.position(season),
            player,
            player.team(1, season),
            price / 10 if price is not None else None,
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
                    pos,
                    opt_squad.tag,
                    purchase_price=opt_squad.dummy_sub_cost,
                )
                squad.add_player(dp)
                logger.debug("%s %s %s", dp.position, dp.name, dp.purchase_price / 10)

    logger.debug("£%sm in the bank", squad.budget / 10)

    return squad
