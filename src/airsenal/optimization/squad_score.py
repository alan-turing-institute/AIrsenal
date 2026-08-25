"""
What a squad is worth over the gameweeks ahead, and how to weigh it.

Named for the squad rather than for scoring, because core/scoring.py already
holds FPL's own points rules and the two are not the same subject: that one says
what a goal is worth, this one says what a squad full of predicted goals is
worth to a search that has to compare next week against five weeks out.

`SquadScoringConfig` lives here rather than in a config module because this is
where it is read. Every optimizer is handed one, so that the squad builder and
the transfer search cannot weigh a bench differently - which is what they did
for as long as the settings were default arguments repeated across both.
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

    `discount ** n_ahead`. There used to be a "constant" alternative
    (`1 - (1 - discount) * n_ahead`, hitting zero fifteen weeks out) selected by
    a string argument, but nothing outside its own test ever passed one.
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

        if gw != bench_boost_gw and sub_weights is not None:
            total_points += gw_weight * squad.total_points_for_subs(
                gw, tag, sub_weights=sub_weights
            )

    return total_points
