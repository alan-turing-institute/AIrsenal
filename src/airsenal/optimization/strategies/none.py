"""Make no transfers at all - the baseline every other strategy is judged against."""

from airsenal.optimization.moves import GameweekMove
from airsenal.optimization.protocols import Proposal, TransferRequest


class NoTransfersStrategy:
    """Keep the squad as it is."""

    def num_increments(self, move: GameweekMove, num_iterations: int) -> int:  # noqa: ARG002
        return 1

    def propose(self, request: TransferRequest) -> Proposal:
        request.advance_progress()
        return Proposal(request.squad, [], [])
