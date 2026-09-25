"""Make no transfers at all - the baseline every other strategy is judged against."""

from airsenal.optimization.protocols import Proposal, TransferRequest


class NoTransfersStrategy:
    """Keep the squad as it is."""

    def num_increments(self, request: TransferRequest) -> int:  # noqa: ARG002
        return 1

    def propose(self, request: TransferRequest) -> Proposal:
        request.advance_progress()
        return Proposal(request.squad, [], [])
