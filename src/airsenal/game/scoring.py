"""What FPL awards for goals, assists, clean sheets, appearances and the rest."""

from airsenal.game.enums import Position

# Keyed by Position so a typo is an AttributeError rather than a KeyError at
# run time, and annotated `str` because what indexes them is a position read
# off a database row. Position is a StrEnum, so the two are the same key.
points_for_goal: dict[str, int] = {
    Position.GK: 10,
    Position.DEF: 6,
    Position.MID: 5,
    Position.FWD: 4,
}

points_for_cs: dict[str, int] = {
    Position.GK: 4,
    Position.DEF: 4,
    Position.MID: 1,
    Position.FWD: 0,
}

points_for_assist = 3

points_for_yellow_card = -1

points_for_red_card = -3

points_for_own_goal = -2

saves_for_point = 3

def_cons_required: dict[str, int] = {
    Position.GK: 999,
    Position.DEF: 10,
    Position.MID: 12,
    Position.FWD: 12,
}

points_for_def_cons = 2


def get_appearance_points(minutes: float) -> float:
    """Points for appearing, and more for playing most of the match."""
    app_points = 0.0
    if minutes > 0:
        app_points = 1
        if minutes >= 60:
            app_points += 1
    return app_points


# Match and modelling limits. These live here rather than with the prediction
# code because the database query layer needs them too, and db must not depend
# on prediction.
MAX_GOALS = 10
MIN_MINUTES_SHORT = 30
MIN_MINUTES_FULL = 60
MAX_MINUTES_MATCH = 90


# Squad and transfer rules. Here for the same reason as the limits above: they
# are FPL's own numbers rather than anything about how we search. Anything that
# applies them to a `GameweekMove` stays in `optimization/moves.py`, because
# `game` cannot depend on optimization - but the arithmetic itself is a rule of
# the game, and lives here so the search and the entry's own state cannot
# disagree about it.
SQUAD_SIZE = 15
MAX_FREE_TRANSFERS = 5  # changed in 24/25 season (not accounted for in replay season)
POINTS_HIT_COST = 4  # points lost per transfer beyond the free ones


def free_transfers_after(
    n_transfers: int,
    prev_free_transfers: int,
    max_free_transfers: int = MAX_FREE_TRANSFERS,
    rebuilds_squad: bool = False,
) -> int:
    """
    How many free transfers are left the gameweek after `n_transfers` were made.

    One is added per week, capped at `max_free_transfers`, and the result is never
    below 1.

    Args:
        rebuilds_squad: True for a wildcard or free hit, which leave the count
            untouched rather than consuming or accruing anything.
    """
    if rebuilds_squad:
        return prev_free_transfers
    return max(1, min(max_free_transfers, 1 + prev_free_transfers - n_transfers))
