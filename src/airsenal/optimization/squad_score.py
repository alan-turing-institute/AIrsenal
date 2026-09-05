"""What a squad is worth over the gameweeks ahead, and how to weigh it."""

from dataclasses import dataclass, field

from airsenal.squad.squad import Squad, SubWeights

DEFAULT_DISCOUNT = 14 / 15  # weight applied per gameweek into the future


@dataclass(frozen=True)
class SquadScoringConfig:
    """How a squad is scored during optimisation."""

    sub_weights: SubWeights = field(default_factory=SubWeights)
    # What a placeholder costs while a partial squad is being filled
    dummy_sub_cost: int = 45
    budget: int = 1000


def get_discount_factor(
    next_gw: int, pred_gw: int, discount: float = DEFAULT_DISCOUNT
) -> float:
    """
    How much a gameweek `pred_gw - next_gw` weeks out counts towards a score.

    `discount ** n_ahead`, so the weight decays geometrically and never reaches
    zero.
    """
    return discount ** (pred_gw - next_gw)


def get_discounted_squad_score(
    squad: Squad,
    gameweeks: list[int],
    tag: str,
    root_gw: int | None = None,
    bench_boost_gw: int | None = None,
    triple_captain_gw: int | None = None,
    *,
    sub_weights: SubWeights,
) -> float:
    """
    Points a squad is expected to score across `gameweeks`, discounted.

    Gameweeks further from `root_gw` count for less; see `get_discount_factor`.
    `root_gw` defaults to the first gameweek in the list.

    Args:
        sub_weights: How much the bench counts outside a bench-boost gameweek.
            `SubWeights.none()` to ignore it.
    """
    if root_gw is None:
        root_gw = gameweeks[0]
    total_points = 0.0
    for gw in gameweeks:
        gw_weight = get_discount_factor(root_gw, gw)
        if gw == bench_boost_gw:
            total_points += (
                squad.get_expected_points(tag, gw, bench_boost=True) * gw_weight
            )
        elif gw == triple_captain_gw:
            total_points += (
                squad.get_expected_points(tag, gw, triple_captain=True) * gw_weight
            )
        else:
            total_points += squad.get_expected_points(tag, gw) * gw_weight

        if gw != bench_boost_gw:
            total_points += gw_weight * squad.total_points_for_subs(
                tag,
                gw,
                sub_weights=sub_weights,
            )

    return total_points
