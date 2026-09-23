"""The pipeline settings that several commands build from the same flags."""

from airsenal.optimization.moves import ChipGameweeks
from airsenal.optimization.protocols import TransferConstraints
from airsenal.optimization.squad_score import SquadScoringConfig
from airsenal.squad.squad import SubWeights


def chip_gameweeks(
    wildcard_gameweek: int,
    free_hit_gameweek: int,
    triple_captain_gameweek: int,
    bench_boost_gameweek: int,
) -> ChipGameweeks:
    return ChipGameweeks(
        wildcard=wildcard_gameweek,
        free_hit=free_hit_gameweek,
        triple_captain=triple_captain_gameweek,
        bench_boost=bench_boost_gameweek,
    )


def transfer_constraints(
    max_hit: int, allow_unused: bool, max_transfers: int
) -> TransferConstraints:
    return TransferConstraints(
        max_total_hit=max_hit,
        allow_unused_transfers=allow_unused,
        max_opt_transfers=max_transfers,
    )


def squad_scoring(
    subs: bool, budget: int = SquadScoringConfig.budget
) -> SquadScoringConfig:
    """Scoring that counts substitutes' points only if `subs` is set."""
    return SquadScoringConfig(
        sub_weights=SubWeights() if subs else SubWeights.none(), budget=budget
    )
