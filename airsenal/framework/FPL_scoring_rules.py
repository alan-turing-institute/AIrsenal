"""
How many points does FPL assign for goals, assists, clean sheets, appearances
"""

points_for_goal = {"GK": 6, "DEF": 6, "MID": 5, "FWD": 4}

points_for_cs = {"GK": 4, "DEF": 4, "MID": 1, "FWD": 0}

points_for_assist = 3

points_for_yellow_card = -1

points_for_red_card = -3

points_for_own_goal = -2

saves_for_point = 3


def get_appearance_points(minutes):
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
