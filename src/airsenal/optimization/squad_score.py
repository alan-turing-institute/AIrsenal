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
    root_gameweek: int, gameweek: int, discount: float = DEFAULT_DISCOUNT
) -> float:
    """
    How much a gameweek counts towards a score, given how far ahead of the root it is.

    `discount ** n_ahead`, so the weight decays geometrically and never reaches
    zero.
    """
    return discount ** (gameweek - root_gameweek)


def get_discounted_squad_score(
    squad: Squad,
    gameweeks: list[int],
    tag: str,
    root_gameweek: int | None = None,
    bench_boost_gameweek: int | None = None,
    triple_captain_gameweek: int | None = None,
    *,
    sub_weights: SubWeights,
) -> float:
    """
    Points a squad is expected to score across `gameweeks`, discounted.

    Gameweeks further from `root_gameweek` count for less; see `get_discount_factor`.
    `root_gameweek` defaults to the first gameweek in the list.

    Args:
        sub_weights: How much the bench counts outside a bench-boost gameweek.
            `SubWeights.none()` to ignore it.
    """
    if root_gameweek is None:
        root_gameweek = gameweeks[0]
    total_points = 0.0
    for gameweek in gameweeks:
        gameweek_weight = get_discount_factor(root_gameweek, gameweek)
        if gameweek == bench_boost_gameweek:
            total_points += (
                squad.get_expected_points(tag, gameweek, bench_boost=True)
                * gameweek_weight
            )
        elif gameweek == triple_captain_gameweek:
            total_points += (
                squad.get_expected_points(tag, gameweek, triple_captain=True)
                * gameweek_weight
            )
        else:
            total_points += squad.get_expected_points(tag, gameweek) * gameweek_weight

        if gameweek != bench_boost_gameweek:
            total_points += gameweek_weight * squad.total_points_for_subs(
                tag,
                gameweek,
                sub_weights=sub_weights,
            )

    return total_points
