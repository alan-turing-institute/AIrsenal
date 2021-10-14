import pygmo as pg

from airsenal.framework.optimization_pygmo import make_new_squad_pygmo
from airsenal.framework.squad import TOTAL_PER_POSITION
from airsenal.framework.utils import (
    CURRENT_SEASON,
    NEXT_GAMEWEEK,
    get_latest_prediction_tag,
)

# -------------------------
#         INPUTS
# -------------------------
# Gameweek range and budget
gw_start = NEXT_GAMEWEEK
num_gw = 3
season = CURRENT_SEASON
tag = get_latest_prediction_tag()
budget = 1000

# Weighting given to substitutes (GK, and 1st, 2nd, 3rd outfield sub).
# 1 means points scored by subs treated as important as points scored by the first XI
# 0 means points scored by subs not considered in optimisation
sub_weights = {"GK": 0.01, "Outfield": (0.4, 0.1, 0.02)}

# Can choose not to optimize full squad by changing the number of players per position.
# In that case, to create valid squads during the optimization, the empty slots are
# filled with dummy players. Each dummy player has a cost of 'dummy_sub_cost'.
players_per_position = TOTAL_PER_POSITION
dummy_sub_cost = 45

# Optimisation algorithm
# For the list of algorithms available in pygmo see here:
# https://esa.github.io/pygmo2/overview.html#list-of-algorithms
# Time it will take to run normally controlled by:
#  - "gen" argument of algorithm - no. of generations (no. of times population
#    is evolved)
#  - "population_size" - no. of candidate solutions in each generatino
uda = pg.sga(gen=100)  # ("User Defined Algorithm")
population_size = 100

# -------------------------
#     RUN OPTIMIZATION
# -------------------------
gw_range = list(range(gw_start, min(38, gw_start + num_gw)))

team = make_new_squad_pygmo(
    gw_range,
    tag,
    uda=uda,
    population_size=population_size,
    sub_weights=sub_weights,
    budget=budget,
    players_per_position=players_per_position,
    dummy_sub_cost=dummy_sub_cost,
    season=season,
)

print("=" * 10)
pts = team.get_expected_points(gw_range[0], tag)
print(f"GW{gw_range[0]}: {pts:.0f} pts")
print(team)
