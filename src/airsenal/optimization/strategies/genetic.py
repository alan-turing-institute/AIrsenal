"""
Replace three or more players with a genetic algorithm.

There are too many combinations to enumerate once more than two players change.
Rather than sampling them at random, this breeds squads within the transfer
limit of the current one, as a wildcard breeds whole squads.
"""

from airsenal.game.enums import Position
from airsenal.optimization.protocols import Proposal, TransferRequest
from airsenal.optimization.squad_optimizers import GeneticAlgorithmConfig
from airsenal.optimization.squad_optimizers.genetic_algorithm import SquadOpt
from airsenal.squad.player import SquadPlayer


def _position_order(player: SquadPlayer) -> int:
    return Position.back_to_front().index(Position(player.position))


class GeneticTransferStrategy:
    """
    Search for the best squad at most `move.n_transfers` changes away.

    It may make fewer changes than the move allows. The hit is still charged for
    all of them, so the smaller move, which the tree also tries, scores better.
    """

    def __init__(self, config: GeneticAlgorithmConfig | None = None) -> None:
        self.config = config if config is not None else GeneticAlgorithmConfig()

    def num_increments(self, request: TransferRequest) -> int:
        return self.config.scaled(request.num_iterations).generations

    def propose(self, request: TransferRequest) -> Proposal:
        squad = request.squad
        opt = SquadOpt(
            request.gameweeks,
            request.tag,
            root_gameweek=request.root_gameweek,
            season=request.season,
            bench_boost_gameweek=request.bench_boost_gameweek,
            triple_captain_gameweek=request.triple_captain_gameweek,
            scoring=request.scoring,
            base_squad=squad,
            max_transfers=request.move.n_transfers,
        )
        best_individual, _ = opt.optimize(
            self.config.scaled(request.num_iterations),
            on_generation=lambda _best_score: request.advance_progress(),
        )
        new_squad = opt.build_squad(best_individual)
        if new_squad is None:
            msg = "The genetic transfer search returned an illegal squad"
            raise RuntimeError(msg)

        before = {p.player_id for p in squad.players}
        after = {p.player_id for p in new_squad.players}
        # in the same order by position, so each sale pairs with its replacement
        players_out = sorted(
            (p for p in squad.players if p.player_id not in after),
            key=_position_order,
        )
        players_in = sorted(
            (p for p in new_squad.players if p.player_id not in before),
            key=_position_order,
        )
        return Proposal(
            new_squad,
            players_in=[p.player_id for p in players_in],
            players_out=[p.player_id for p in players_out],
        )
