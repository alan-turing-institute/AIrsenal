"""
How many points does FPL assign for goals, assists, clean sheets, appearances
"""

points_for_goal = {"GK": 10, "DEF": 6, "MID": 5, "FWD": 4}

points_for_cs = {"GK": 4, "DEF": 4, "MID": 1, "FWD": 0}

points_for_assist = 3

points_for_yellow_card = -1

points_for_red_card = -3

points_for_own_goal = -2

saves_for_point = 3

def_cons_required = {"GK": 999, "DEF": 10, "MID": 12, "FWD": 12}

points_for_def_cons = 2


def get_appearance_points(minutes: float) -> float:
    """
    get points for being on the pitch at all, and more for being on
    for most of the match.
    """
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
