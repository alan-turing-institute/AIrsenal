"""
The branches of a transfer plan tree, and what taking one of them scores.

Shared by every transfer optimizer that walks the tree one gameweek at a time:
`next_gameweek_transfers` says which moves are legal next, `make_best_transfers`
makes one and scores it, and `count_expected_outputs` sizes the whole tree.
"""

from collections.abc import Iterable

from airsenal.db.queries.gameweeks import next_gameweek
from airsenal.game.chips import chips_used_up
from airsenal.game.enums import Chip
from airsenal.game.scoring import MAX_FREE_TRANSFERS
from airsenal.game.season import CURRENT_SEASON
from airsenal.optimization.moves import (
    NO_CHIPS,
    ChipSchedule,
    GameweekChips,
    GameweekMove,
    calc_free_transfers,
    calc_points_hit,
)
from airsenal.optimization.protocols import (
    DEFAULT_MAX_OPT_TRANSFERS,
    Proposal,
    TransferRequest,
    TransferStrategy,
)
from airsenal.optimization.squad_score import get_discounted_squad_score
from airsenal.squad.squad import Squad


def transfer_counts(
    free_transfers: int,
    fewest: int,
    most: int,
    *,
    every_transfer_count: bool = True,
) -> list[int]:
    """
    How many transfers to consider making in a gameweek, from `fewest` to `most`.

    Every count between them, or with `every_transfer_count` False only 0, 1, 2
    and `free_transfers`. The two small counts are searched exhaustively and
    cheaply; the counts between 2 and every free transfer each cost a genetic
    search and multiply the tree, and are rarely the best use of a bank of
    transfers.
    """
    if every_transfer_count:
        counts = set(range(fewest, most + 1))
    else:
        counts = {0, 1, 2, free_transfers}
    return sorted(n for n in counts if fewest <= n <= most)


def next_gameweek_transfers(
    free_transfers: int,
    hit_so_far: int,
    chips_played: Iterable[Chip | None] = (),
    max_total_hit: int | None = None,
    allow_unused_transfers: bool = False,
    max_opt_transfers: int = DEFAULT_MAX_OPT_TRANSFERS,
    chips: GameweekChips | None = None,
    max_free_transfers: int = MAX_FREE_TRANSFERS,
    *,
    every_transfer_count: bool = True,
) -> list[tuple[GameweekMove, int, int, int]]:
    """
    The moves - transfers, and any chip played - a strategy may make next gameweek.

    Args:
        free_transfers: Available going into the gameweek.
        hit_so_far: Points hit this strategy has taken up to but not including
            this gameweek.
        chips_played: Chips this strategy has none of left to play this
            gameweek, so they are not offered; see `game.chips.chips_used_up`.
        allow_unused_transfers: If False and a free transfer would otherwise be
            lost, making none is not offered - which can exclude the baseline
            strategy, so a caller that needs it re-adds it.
        every_transfer_count: If False, only 0, 1 and 2 transfers and one that
            uses every free transfer are offered, of those up to
            `max_opt_transfers`. See `transfer_counts`.

    Returns:
        Per move: the move, the free transfers it leaves for the gameweek after,
        the total hit including this gameweek, and the hit this gameweek alone.
    """
    chips = chips if chips is not None else NO_CHIPS
    chips_played = list(chips_played)

    # Force at least one transfer if a free transfer would otherwise be lost.
    fewest = (
        1 if not allow_unused_transfers and free_transfers == max_free_transfers else 0
    )
    ft_choices = transfer_counts(
        free_transfers,
        fewest,
        max_opt_transfers,
        every_transfer_count=every_transfer_count,
    )

    if max_total_hit is not None:
        ft_choices = [
            nt
            for nt in ft_choices
            if hit_so_far + calc_points_hit(GameweekMove(nt), free_transfers)
            <= max_total_hit
        ]

    # if we are definitely going to play a wildcard or free_hit deal with that first
    if chips.chip_to_play is not None and chips.chip_to_play.rebuilds_squad:
        moves = [GameweekMove(chip=chips.chip_to_play)]
    elif chips.chip_to_play is not None:
        # triple captain or bench boost - we can still do ft_choices transfers
        moves = [GameweekMove(nt, chips.chip_to_play) for nt in ft_choices]
    else:
        # no chip definitely played, but some might be allowed
        moves = [GameweekMove(nt) for nt in ft_choices]
        for chip in (Chip.WILDCARD, Chip.FREE_HIT):
            if chips.allows(chip, chips_played):
                moves.append(GameweekMove(chip=chip))
        for chip in (Chip.BENCH_BOOST, Chip.TRIPLE_CAPTAIN):
            if chips.allows(chip, chips_played):
                moves += [GameweekMove(nt, chip) for nt in ft_choices]

    hit_this_gameweek = [calc_points_hit(move, free_transfers) for move in moves]
    total_points_hit = [hit_so_far + hit for hit in hit_this_gameweek]
    new_ft_available = [
        calc_free_transfers(move, free_transfers, max_free_transfers) for move in moves
    ]

    return list(
        zip(moves, new_ft_available, total_points_hit, hit_this_gameweek, strict=True)
    )


def count_expected_outputs(
    n_gameweeks: int,
    gameweek: int | None = None,
    free_transfers: int = 1,
    max_total_hit: int | None = None,
    allow_unused_transfers: bool = False,
    max_opt_transfers: int = DEFAULT_MAX_OPT_TRANSFERS,
    chip_schedule: ChipSchedule | None = None,
    max_free_transfers: int = MAX_FREE_TRANSFERS,
    *,
    season: str = CURRENT_SEASON,
    chips_played: Iterable[tuple[int, Chip]] = (),
    every_transfer_count: bool = True,
) -> tuple[int, bool]:
    """
    Count the strategies a search over `n_gameweeks` gameweeks will visit.

    Args:
        max_total_hit: Points that may be spent on transfers across the whole
            window; None for no limit.
        allow_unused_transfers: If False, strategies that leave a free transfer
            unused - making none with a full bank of free transfers - are not counted.
        chips_played: (gameweek, chip) for each chip played before the window.

    Returns:
        How many strategies will be computed, and whether the baseline strategy
        falls outside the main tree and so has to be computed separately, which
        `allow_unused_transfers=False` can cause. The count includes the
        baseline either way.
    """
    gameweek = next_gameweek() if gameweek is None else gameweek
    chip_schedule = chip_schedule if chip_schedule is not None else ChipSchedule()
    chips_played = tuple(chips_played)

    # (free transfers, points hit so far, moves made) - the moves are all that is
    # needed to count branches and to spot the do-nothing baseline among them
    branches: list[tuple[int, int, tuple[GameweekMove, ...]]] = [
        (free_transfers, 0, ())
    ]

    for window_gameweek in range(gameweek, gameweek + n_gameweeks):
        new_branches = []
        for ft, hit, moves in branches:
            possibilities = next_gameweek_transfers(
                ft,
                hit,
                chips_used_up(
                    (
                        *chips_played,
                        *zip(
                            range(gameweek, window_gameweek),
                            [move.chip for move in moves],
                            strict=True,
                        ),
                    ),
                    window_gameweek,
                    season,
                ),
                max_total_hit=max_total_hit,
                max_opt_transfers=max_opt_transfers,
                allow_unused_transfers=allow_unused_transfers,
                chips=chip_schedule.for_gameweek(window_gameweek),
                max_free_transfers=max_free_transfers,
                every_transfer_count=every_transfer_count,
            )
            new_branches += [
                (new_ft, new_hit, (*moves, move))
                for move, new_ft, new_hit, _ in possibilities
            ]
        branches = new_branches

    # allow_unused_transfers=False can remove the baseline above, so add it back if
    # the first strategy is not it. `branches` is empty when the constraints admit
    # no move at all: --max-transfers 0, unused transfers disallowed and a full bank
    # of free transfers forces at least one transfer and then allows none.
    baseline_moves = (GameweekMove(),) * n_gameweeks
    baseline_excluded = not branches or branches[0][2] != baseline_moves
    if baseline_excluded:
        branches.insert(0, (max_free_transfers, 0, baseline_moves))

    return len(branches), baseline_excluded


def make_best_transfers(
    request: TransferRequest, strategy: TransferStrategy
) -> tuple[Squad, Proposal, float]:
    """
    Make this gameweek's move and score the squad it leaves. One node of the tree.

    Returns the squad that carries on to the next gameweek, the strategy's
    proposal, and the points the proposed squad is expected to score next gameweek.
    """
    proposal = strategy.propose(request)

    points = get_discounted_squad_score(
        proposal.squad,
        [request.transfer_gameweek],
        request.tag,
        root_gameweek=request.root_gameweek,
        bench_boost_gameweek=request.bench_boost_gameweek,
        triple_captain_gameweek=request.triple_captain_gameweek,
        sub_weights=request.scoring.sub_weights,
    )

    # A free hit is reverted after the gameweek it is played in, so the squad
    # that carries on to the next gameweek is the one we started with.
    resulting_squad = proposal.squad if request.move.carry_forward else request.squad
    return resulting_squad, proposal, points
