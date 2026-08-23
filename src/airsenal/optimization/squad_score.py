"""
What a squad is worth over the gameweeks ahead.

Named for the squad rather than for scoring, because core/scoring.py already
holds FPL's own points rules and the two are not the same subject: that one says
what a goal is worth, this one says what a squad full of predicted goals is
worth to a search that has to compare next week against five weeks out.
"""

from airsenal.optimization.config import SubWeightsDict
from airsenal.squad.squad import Squad

DEFAULT_DISCOUNT = 14 / 15  # weight applied per gameweek into the future


def get_discount_factor(
    next_gw: int,
    pred_gw: int,
    discount_type: str = "exp",
    discount: float = DEFAULT_DISCOUNT,
) -> float:
    """
    given the next gw and a predicted gw, retrieve discount factor. Either:
        - exp: discount**n_ahead (discount reduces each gameweek)
        - const: 1-(1-discount)*n_ahead (constant discount each gameweek, goes to
          zero at gw 15 with default discount)
    """
    allowed_types = ["exp", "const", "constant"]
    if discount_type not in allowed_types:
        msg = "unrecognised discount type, should be exp or const"
        raise Exception(msg)

    n_ahead = pred_gw - next_gw

    if discount_type == "exp":
        return discount**n_ahead
    return max(1 - (1 - discount) * n_ahead, 0)


def get_discounted_squad_score(
    squad: Squad,
    gameweeks: list[int],
    tag: str,
    root_gw: int | None = None,
    bench_boost_gw: int | None = None,
    triple_captain_gw: int | None = None,
    sub_weights: SubWeightsDict | None = None,
) -> float:
    """Get the number of points a squad is expected to score across a number of
    gameweeks, discounting the weight of gameweeks further into the future with respect
    to the root_gw.
    """
    if root_gw is None:
        root_gw = gameweeks[0]
    total_points = 0.0
    for gw in gameweeks:
        gw_weight = get_discount_factor(root_gw, gw)
        if gw == bench_boost_gw:
            total_points += (
                squad.get_expected_points(gw, tag, bench_boost=True) * gw_weight
            )
        elif gw == triple_captain_gw:
            total_points += (
                squad.get_expected_points(gw, tag, triple_captain=True) * gw_weight
            )
        else:
            total_points += squad.get_expected_points(gw, tag) * gw_weight

        if gw != bench_boost_gw and sub_weights:
            total_points += gw_weight * squad.total_points_for_subs(
                gw, tag, sub_weights=sub_weights
            )

    return total_points
