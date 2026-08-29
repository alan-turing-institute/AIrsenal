"""
What a squad is worth over the gameweeks ahead, and how to weigh it.

Named for the squad rather than for scoring, because `game/scoring.py` holds
FPL's own points rules and the two are not the same subject: that one says what a
goal is worth, this one says what a squad full of predicted goals is worth to a
search comparing next week against five weeks out.

Every optimizer is handed the same `SquadScoringConfig`, so the squad builder
and the transfer search cannot weigh a bench differently.
"""

from dataclasses import dataclass, field

from airsenal.squad.squad import Squad, SubWeights

DEFAULT_DISCOUNT = 14 / 15  # weight applied per gameweek into the future


@dataclass(frozen=True)
class SquadScoringConfig:
    """How a squad is scored during optimisation."""

    sub_weights: SubWeights = field(default_factory=SubWeights)
    # What a placeholder costs while a partial squad is being filled. Only bites
    # when `players_per_position` is smaller than a full squad, which nothing but
    # the tests does, so it is effectively fixed - kept a field because the squad
    # builder takes it as one, not because it is a knob anyone turns.
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
    sub_weights: SubWeights | None = None,
) -> float:
    """
    Points a squad is expected to score across `gameweeks`, discounted.

    Gameweeks further from `root_gw` count for less; see `get_discount_factor`.
    `root_gw` defaults to the first gameweek in the list.
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

        if gw != bench_boost_gw and sub_weights is not None:
            total_points += gw_weight * squad.total_points_for_subs(
                tag,
                gw,
                sub_weights=sub_weights,
            )

    return total_points
