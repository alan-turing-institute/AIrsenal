"""
Breaking a realised FPL score into the components that predict it.

The observable side of a component: what a player actually earned from each
part of their performance, so a component's expected points have something to
be scored against. Checked on the live database, this reconstructs
`PlayerScore.points` exactly - for every performance in 2425 and 2526 - which
is what makes it a usable ground truth and a guardrail on the scoring rules.
"""

from airsenal.db.models import PlayerScore
from airsenal.game.enums import Position
from airsenal.game.scoring import (
    MIN_MINUTES_FULL,
    def_cons_required,
    get_appearance_points,
    points_for_assist,
    points_for_cs,
    points_for_def_cons,
    points_for_goal,
    points_for_own_goal,
    points_for_penalty_miss,
    points_for_penalty_save,
    points_for_red_card,
    points_for_yellow_card,
    saves_for_point,
)

# Own goals and penalties: real points that no component predicts. They are
# about 0.5% of all points awarded, so they are reported as their own bucket
# rather than blamed on a component that did not earn them.
RESIDUAL = "residual"


def actual_component_points(score: PlayerScore, position: str) -> dict[str, float]:
    """
    What each component of a score actually came to, for one performance.

    Keyed by the same names the components use, plus `RESIDUAL`. Sums to
    `score.points`.

    Raises:
        ValueError: For a position AIrsenal does not model - `MNG`, the manager
            FPL ran in 24/25 - which has no scoring rules here at all.
    """
    if position not in points_for_goal:
        msg = f"{position} is not a position with scoring rules"
        raise ValueError(msg)
    minutes = score.minutes
    is_defensive = position in (Position.GK, Position.DEF)
    components = {
        "appearance": get_appearance_points(minutes),
        "attacking": float(
            points_for_goal[position] * (score.goals or 0)
            + points_for_assist * (score.assists or 0)
        ),
        "defending": float(
            (
                points_for_cs[position]
                if score.clean_sheets and minutes >= MIN_MINUTES_FULL
                else 0
            )
            - ((score.conceded or 0) // 2 if is_defensive else 0)
        ),
        "bonus": float(score.bonus or 0),
        "cards": float(
            points_for_yellow_card * (score.yellow_cards or 0)
            + points_for_red_card * (score.red_cards or 0)
        ),
        "saves": float(
            (score.saves or 0) // saves_for_point if position == Position.GK else 0
        ),
        RESIDUAL: float(
            points_for_own_goal * (score.own_goals or 0)
            + points_for_penalty_save * (score.penalties_saved or 0)
            + points_for_penalty_miss * (score.penalties_missed or 0)
        ),
    }
    # Only from 25/26, and only recorded for the seasons that have it: a
    # performance from before then has no defensive-contribution component at
    # all rather than one worth zero.
    if score.defensive_contribution is not None:
        components["def_con"] = float(
            points_for_def_cons
            if score.defensive_contribution >= def_cons_required[position]
            else 0
        )
    return components
