"""
Simple approximation of the probablity of each substitute being used.
The output of this is used as the default sub weighting in
airsenal.framework.optimization_pygmo.TeamOpt

TODO: Determine the actual probability of an XI player not playing
TODO: Use different probabilities for each position (GK, DEF, MID, FWD)
TODO: Take into account different formations and sub (position) orders
"""
import numpy as np

# number of random trials
n_samples = int(1e6)

# probability that an outfield player in the XI plays at least 1 minute
prob_play = 0.95

# 10 outfield players + 3 subs
is_playing = np.random.uniform(size=(n_samples, 13)) < prob_play

# 1st sub plays if at least 1 of first 10 players doesn't play
p_sub_1 = (is_playing[:, :10].sum(axis=1) < 10).mean()
print("Sub 1:", p_sub_1)

# 2nd sub plays if at least 2 of first 11 players
# (starting 10 outfield + first sub) don't play
p_sub_2 = (is_playing[:, :11].sum(axis=1) < 10).mean()
print("Sub 2:", p_sub_2)

# 3rd sub plays if at least 3 of first 12 players
# (starting 10 outfield + first 2 subs) don't play
p_sub_3 = (is_playing[:, :12].sum(axis=1) < 10).mean()
print("Sub 3:", p_sub_3)
