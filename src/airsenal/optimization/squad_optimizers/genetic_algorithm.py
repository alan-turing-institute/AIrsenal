"""The DEAP genetic algorithm itself: pick a whole squad, generation by generation."""

import copy
import random
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace

import numpy as np
from deap import algorithms, base, creator, tools
from sqlalchemy.orm import Session

from airsenal.core.logging import get_logger
from airsenal.db.models import Player
from airsenal.db.queries.players import list_players
from airsenal.db.queries.predictions import get_predicted_points_for_player
from airsenal.game.enums import Position
from airsenal.game.scoring import SQUAD_SIZE
from airsenal.game.season import CURRENT_SEASON
from airsenal.optimization.squad_score import (
    SquadScoringConfig,
    get_discounted_squad_score,
)
from airsenal.squad.player import CandidateCache, DummyPlayer
from airsenal.squad.squad import TOTAL_PER_POSITION, Squad

logger = get_logger(__name__)

# What an illegal squad scores when making transfers: below anything a legal squad
# can score, so the two never tie, but finite, so DEAP's statistics stay defined.
ILLEGAL_TRANSFER_FITNESS = -1e9


@dataclass(frozen=True)
class GeneticAlgorithmConfig:
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
        """Scale the population and generations from a single number."""
        return replace(self, population_size=num_iterations, generations=num_iterations)


# Called after each generation, with the best fitness found so far.
type GenerationReporter = Callable[[float], None]


def _ensure_deap_types() -> None:
    """
    Register the DEAP fitness and individual classes, once per process.

    `creator.create` writes module-level state, and warns if a class is created twice.
    """
    if not hasattr(creator, "AirsenalFitnessMax"):
        creator.create("AirsenalFitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "AirsenalIndividual"):
        creator.create("AirsenalIndividual", list, fitness=creator.AirsenalFitnessMax)


class SquadOpt:
    """
    DEAP-based optimization of a fantasy football squad.

    Args:
        scoring: The budget, the bench weighting and what a placeholder costs -
            see `SquadScoringConfig`. Required rather than defaulted, so that a
            squad is never scored with a bench weighting nobody chose.
        players_per_position: How many players to optimize in each position.
            Anything short of a full squad leaves the rest as dummies.
        remove_zero: If True, players with a predicted total of zero points are
            not considered at all.
        root_gameweek: The gameweek every price and club is read as at, and the
            origin the discount decays from. Defaults to the first of `gameweeks`;
            see `optimization.protocols.SquadRequest.root_gameweek`.
        base_squad: A squad to make transfers from, rather than building one from
            nothing. An individual is then the squad after at most
            `max_transfers` of its players are replaced: the rest keep what was
            paid for them, and each sale adds its sell price to the bank.
            `scoring.budget` is not used, and every position is optimised.
    """

    def __init__(
        self,
        gameweeks: list[int],
        tag: str,
        root_gameweek: int | None = None,
        season: str = CURRENT_SEASON,
        bench_boost_gameweek: int | None = None,
        triple_captain_gameweek: int | None = None,
        remove_zero: bool = True,
        players_per_position: dict[str, int] = TOTAL_PER_POSITION,
        *,
        scoring: SquadScoringConfig,
        base_squad: Squad | None = None,
        max_transfers: int | None = None,
        dbsession: Session | None = None,
    ) -> None:
        if (base_squad is None) != (max_transfers is None):
            msg = "base_squad and max_transfers are given together or not at all"
            raise ValueError(msg)
        self.dbsession = dbsession
        self.season = season
        self.gameweeks = gameweeks
        self.root_gameweek = (
            root_gameweek if root_gameweek is not None else min(gameweeks)
        )
        self.bench_boost_gameweek = bench_boost_gameweek
        self.triple_captain_gameweek = triple_captain_gameweek

        self.tag = tag
        self.positions = list(Position.back_to_front())
        self.players_per_position = players_per_position
        self.n_opt_players = sum(self.players_per_position.values())
        # no. players each position that won't be optimised (just filled with dummies)
        self.dummy_per_position = self._get_dummy_per_position()
        self.scoring = scoring

        self.base_squad = base_squad
        self.max_transfers = max_transfers
        # every squad the search builds is priced as at the root gameweek, so
        # each player is built once, however many individuals include them
        self.candidates = CandidateCache(
            self.root_gameweek, base_squad.season if base_squad else season
        )
        self._base_ids = (
            {p.player_id for p in base_squad.players} if base_squad else set()
        )

        self.players, self.position_idx = self._get_player_list()
        if remove_zero:
            self._remove_zero_pts()
        self.n_available_players = len(self.players)

        # the position each gene holds, in the order individuals are laid out
        self._slot_positions = [
            pos for pos in self.positions for _ in range(players_per_position[pos])
        ]
        self._sale_prices: dict[int, int] = {}
        self._seed: list[int] = []
        if base_squad is not None:
            self._sale_prices = {
                p.player_id: base_squad.get_sell_price_for_player(
                    p, gameweek=self.root_gameweek, dbsession=dbsession
                )
                for p in base_squad.players
            }
            self._seed = self._base_squad_indices()

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

    def _base_squad_indices(self) -> list[int]:
        """
        The base squad as an individual, as far as the candidate pool holds it.

        A player the pool does not have - one no longer listed as at the root
        gameweek - is stood in for by a random player in the same position, which
        counts as one of the transfers.
        """
        assert self.base_squad is not None
        index_of = {p.player_id: i for i, p in enumerate(self.players)}
        by_position: dict[Position, list[int]] = {pos: [] for pos in self.positions}
        for p in self.base_squad.players:
            by_position[Position(p.position)].append(p.player_id)

        indices = []
        for pos in self._slot_positions:
            index = None
            while by_position[pos] and index is None:
                index = index_of.get(by_position[pos].pop())
            if index is None:
                pos_min, pos_max = self.position_idx[pos]
                index = random.randint(pos_min, pos_max)
            indices.append(index)
        return indices

    def _create_individual(self) -> list[int]:
        """
        A random starting squad, as indices into `self.players`.

        The list is grouped by position and each group is drawn from that
        position's contiguous slice, so an individual is always positionally
        valid even before the budget is checked.

        Making transfers, it is the base squad with up to `max_transfers` slots
        redrawn. Drawn from scratch, almost no individual would be within that
        many transfers of the base squad, and the search would have no valid
        squad to start from.
        """
        if self.base_squad is not None:
            assert self.max_transfers is not None
            individual = list(self._seed)
            n_changes = random.randint(0, self.max_transfers)
            for slot in random.sample(range(len(individual)), n_changes):
                pos_min, pos_max = self.position_idx[self._slot_positions[slot]]
                individual[slot] = random.randint(pos_min, pos_max)
            return creator.AirsenalIndividual(individual)

        individual = []
        for pos in self.positions:
            pos_min, pos_max = self.position_idx[pos]
            n_players = self.players_per_position[pos]
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
        for pos in self.positions:
            pos_min, pos_max = self.position_idx[pos]
            n_players = self.players_per_position[pos]
            low_bounds.extend([pos_min] * n_players)
            up_bounds.extend([pos_max] * n_players)

        return low_bounds, up_bounds

    def build_squad(self, individual: list[int]) -> Squad | None:
        """
        The squad an individual stands for, or None if it is not a legal squad.

        Over budget, too many players from one club, a duplicated player, or more
        than `max_transfers` changes to the base squad are all illegal.
        """
        players = [self.players[int(idx)] for idx in individual]
        if self.base_squad is not None:
            return self._transfer_from_base(players)

        squad = Squad(budget=self.scoring.budget, season=self.season)

        for player in players:
            if not self._add(squad, player):
                return None

        for dp in self.dummies():
            if not squad.add_player(dp):
                return None

        return squad if squad.is_complete() else None

    def _add(self, squad: Squad, player: Player) -> bool:
        """Add `player` to `squad` if it fits, from the candidates already built."""
        candidate = self.candidates.get(player)
        if not squad.can_add(candidate):
            return False
        return squad.add_player(copy.copy(candidate), gameweek=self.root_gameweek)

    def _transfer_from_base(self, players: list[Player]) -> Squad | None:
        """The base squad with the transfers that make it `players`."""
        assert self.base_squad is not None
        assert self.max_transfers is not None
        player_ids = [player.player_id for player in players]
        if len(set(player_ids)) != len(player_ids):
            return None
        players_in = [p for p in players if p.player_id not in self._base_ids]
        if len(players_in) > self.max_transfers:
            return None

        squad = self.base_squad.copy()
        for player_id in self._base_ids.difference(player_ids):
            squad.remove_player(
                player_id,
                price=self._sale_prices[player_id],
                gameweek=self.root_gameweek,
                dbsession=self.dbsession,
            )
        for player in players_in:
            if not self._add(squad, player):
                return None
        return squad if squad.is_complete() else None

    def _evaluate_individual(self, individual: list[int]) -> tuple[float]:
        """
        The squad's discounted score, or 0.0 if it is not a legal squad.

        An illegal squad scores zero rather than raising, which is how the GA
        discards it; see `build_squad`. Making transfers, it scores
        `ILLEGAL_TRANSFER_FITNESS` instead, so it can never tie with a legal
        squad that scores nothing.
        """
        squad = self.build_squad(individual)
        if squad is None:
            if self.base_squad is not None:
                return (ILLEGAL_TRANSFER_FITNESS,)
            return (0.0,)

        score = get_discounted_squad_score(
            squad,
            self.gameweeks,
            self.tag,
            self.root_gameweek,
            self.bench_boost_gameweek,
            self.triple_captain_gameweek,
            sub_weights=self.scoring.sub_weights,
        )

        return (score,)

    def _get_player_list(self) -> tuple[list[Player], dict[Position, tuple[int, int]]]:
        """
        The players active as at the root gameweek, and where each position sits.

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
                gameweek=self.root_gameweek,
                dbsession=self.dbsession,
            )
            change_idx.append(len(players))
        return players, self._position_blocks(change_idx)

    def _remove_zero_pts(self) -> None:
        """
        Exclude players with zero predicted points.

        A position left with fewer players than it needs keeps all of them, so
        every position can still fill its slots.
        """
        players: list[Player] = []
        change_idx = [0]
        for pos in self.positions:
            first, last = self.position_idx[pos]
            candidates = self.players[first : last + 1]
            # the base squad's own players stay, so keeping one is always possible
            scoring = [
                p
                for p in candidates
                if p.player_id in self._base_ids or self._total_points(p) > 0
            ]
            if len(scoring) < self.players_per_position[pos]:
                logger.warning(
                    "Only %d %s players have predicted points; considering all %d.",
                    len(scoring),
                    pos,
                    len(candidates),
                )
                scoring = candidates
            players += scoring
            change_idx.append(len(players))

        self.players = players
        self.position_idx = self._position_blocks(change_idx)

    def _total_points(self, player: Player) -> float:
        """A player's predicted points summed over the gameweeks being optimised."""
        gameweek_points = get_predicted_points_for_player(
            player, self.tag, season=self.season, dbsession=self.dbsession
        )
        return sum(
            pts
            for gameweek, pts in gameweek_points.items()
            if gameweek in self.gameweeks
        )

    def _position_blocks(
        self, change_idx: list[int]
    ) -> dict[Position, tuple[int, int]]:
        """The (first, last) index of each position's block, from where blocks start."""
        return {
            self.positions[i - 1]: (change_idx[i - 1], change_idx[i] - 1)
            for i in range(1, len(change_idx))
        }

    def _get_dummy_per_position(self) -> dict[Position, int]:
        """How many dummies each position needs to bring the squad up to full size."""
        return {
            pos: (TOTAL_PER_POSITION[pos] - self.players_per_position[pos])
            for pos in self.positions
        }

    def dummies(self) -> Iterator[DummyPlayer]:
        """Fresh placeholders for every squad slot that is not being optimised."""
        for pos in self.positions:
            for _ in range(self.dummy_per_position[pos]):
                yield DummyPlayer(
                    self.gameweeks,
                    pos,
                    self.tag,
                    purchase_price=self.scoring.dummy_sub_cost,
                )

    def optimize(
        self,
        config: GeneticAlgorithmConfig | None = None,
        on_generation: GenerationReporter | None = None,
    ) -> tuple[list[int], float]:
        """
        Run the genetic algorithm, returning the best individual and its fitness.

        Args:
            on_generation: Called after each generation with the best fitness so
                far. Given one, the search is run a generation at a time so that
                it can report.
        """
        config = config if config is not None else GeneticAlgorithmConfig()
        if config.random_state is not None:
            random.seed(config.random_state)
            np.random.seed(config.random_state)

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
        if self.base_squad is not None:
            # Making no transfers is always legal, so the search always has at
            # least one valid squad to return.
            population[0] = creator.AirsenalIndividual(self._seed)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

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
        if best_fitness <= ILLEGAL_TRANSFER_FITNESS:
            msg = (
                f"The transfer search found no legal squad within "
                f"{self.max_transfers} transfers of the base squad: every one of "
                f"{config.population_size} individuals over {config.generations} "
                "generations broke the budget, the club limit or the transfer "
                "limit."
            )
            raise RuntimeError(msg)

        return best_individual, best_fitness

    def _run_generations(
        self,
        population: list["creator.AirsenalIndividual"],
        config: GeneticAlgorithmConfig,
        stats: tools.Statistics,
        hall_of_fame: tools.HallOfFame,
        on_generation: GenerationReporter,
    ) -> None:
        """Run `eaSimple` optimisation, reporting status after each generation."""
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
    players_per_position: dict[str, int] = TOTAL_PER_POSITION,
    root_gameweek: int | None = None,
    season: str = CURRENT_SEASON,
    bench_boost_gameweek: int | None = None,
    triple_captain_gameweek: int | None = None,
    remove_zero: bool = True,
    *,
    scoring: SquadScoringConfig,
    ga_config: GeneticAlgorithmConfig | None = None,
    on_generation: GenerationReporter | None = None,
    dbsession: Session | None = None,
) -> Squad:
    """
    Optimize a full initial squad using the DEAP genetic algorithm.

    Everything up to `scoring` is passed straight to `SquadOpt`, which documents
    it. Beyond that:

    Args:
        on_generation: Called after each generation with the best score so far,
            for a caller that wants to show progress. An alternative to the
            config's `verbose`, which prints DEAP's own logbook instead.
    """
    opt_squad = SquadOpt(
        gameweeks,
        tag,
        players_per_position=players_per_position,
        root_gameweek=root_gameweek,
        season=season,
        bench_boost_gameweek=bench_boost_gameweek,
        triple_captain_gameweek=triple_captain_gameweek,
        remove_zero=remove_zero,
        scoring=scoring,
        dbsession=dbsession,
    )

    ga_config = ga_config if ga_config is not None else GeneticAlgorithmConfig()
    best_individual, best_fitness = opt_squad.optimize(ga_config, on_generation)

    logger.debug("Best score: %s pts", best_fitness)

    squad = Squad(budget=opt_squad.scoring.budget, season=season)
    for idx in best_individual:
        player = opt_squad.players[int(idx)]
        # as at the gameweek the squad was scored and costed at
        price = player.price(opt_squad.root_gameweek, season)
        logger.debug(
            "%s %s %s %s",
            player.position(season),
            player,
            player.team(opt_squad.root_gameweek, season),
            price / 10 if price is not None else None,
        )
        squad.add_player(
            player.player_id,
            gameweek=opt_squad.root_gameweek,
            dbsession=dbsession,
        )

    for dp in opt_squad.dummies():
        squad.add_player(dp)
        logger.debug("%s %s %s", dp.position, dp.name, dp.purchase_price / 10)

    logger.debug("£%sm in the bank", squad.budget / 10)

    if not squad.is_complete():
        msg = (
            f"The squad search found no legal squad within "
            f"£{scoring.budget / 10}m: the "
            f"best of {ga_config.population_size} individuals over "
            f"{ga_config.generations} generations holds {len(squad.players)} of "
            f"{SQUAD_SIZE} players. Raise the budget, or widen the pool of "
            f"candidates ({opt_squad.n_available_players} players considered)."
        )
        raise ValueError(msg)

    return squad
