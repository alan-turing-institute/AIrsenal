# The xG models: how the defaults were chosen

`XGTeamModel` and `XGPlayerModel` are the models AIrsenal uses by default. This page
records where every number in them came from: how each hyperparameter was chosen, and
which ideas seemed promising but made the models worse. Comments in the source point
here for the reasoning behind `DEFAULT_XG_EPSILON`, `DEFAULT_GOAL_DISPERSION`,
`DEFAULT_XG_N_GOALS_PRIOR`, `DEFAULT_XG_GOAL_WEIGHT` and `promoted_like_bottom`.

To add a model of your own and evaluate it in the same way, see
[adding-a-model.md](adding-a-model.md).

Unless stated otherwise, the scores below are held-out mean log probabilities, where
higher is better. Differences are in nats.

## The xG team model

`XGTeamModel` in `team_models/xg.py` only predicts the mean number of goals each team
will score. Its table entry wraps it in `ConwayMaxwellScorelines` (from
`scorelines.py`) to turn that mean into probabilities for each number of goals. The
model gives each team an attack and a defence rating, fitted to the expected goals in
past matches rather than the actual goals. The fit alternates between attack and
defence ratings, so each team's attack is rated against the defences it actually
played. That separates a good attack from one that has had an easy run of fixtures.

It gets expected goals from `home_expected_goals` and `away_expected_goals`, which are
`NotRequired` keys on `TeamFitData` filled in by `get_result_dict` from
`get_expected_goals_by_fixture`.

All measurements in this section used `conjugate` as the player model. Up to the
Conway-Maxwell-Poisson section, they also used `PoissonScorelines` as the wrapper.

**It beats `extended` in every season in the database.** Held-out mean log
probability over gameweeks 5-30:

| season | `xg` | `extended` | difference |
|---|---|---|---|
| 2324 | -3.07950 | -3.11516 | +0.03566 |
| 2425 | -2.99039 | -3.02760 | +0.03721 |
| 2526 | -2.85431 | -2.87858 | +0.02428 |

For predicted points over 2526 gameweeks 5-30, it's a small improvement on every
measure, and the breakdown by component shows where the improvement comes from:

| metric | `extended` | `xg` |
|---|---|---|
| points MAE | 0.905191 | 0.903423 |
| points MAE, appeared | 2.092136 | 2.088024 |
| points RMSE | 1.883554 | 1.881903 |
| rank correlation | 0.815694 | 0.816195 |
| attacking component MAE | 0.410864 | 0.415394 |
| defending component MAE | 0.272859 | 0.265226 |

**All of the improvement is in defending points, and attacking points get slightly
worse.** xG predicts how many goals a team concedes better, and clean sheet and
goals-conceded points follow directly from that. Which players score a team's goals is
decided by the player model, which hasn't changed. So a better team model on its own
mostly improves predicted points through defending.

### Time weighting

Past matches are weighted by `exp(-epsilon * years ago)`, and `--epsilon` sets
epsilon. Results over 2425 and 2526 from `tools/tune_team_time_weighting.py --model xg`:

| epsilon | 0.0 | 0.3 | 0.6 | 0.9 | 1.2 | 1.8 | 2.5 |
|---|---|---|---|---|---|---|---|
| avg log prob | -2.93330 | -2.93234 | **-2.93224** | -2.93285 | -2.93392 | -2.93674 | -2.94025 |

The best value is only about 0.001 better than no weighting at all. It's the default
because it's the best value, not because it makes much difference. Expected goals vary
less from match to match than goals do: the Dixon-Coles models are best at 0.9 and
gain noticeably from it, whereas this model barely notices. The margin over `extended`
is unchanged: +0.03552, +0.04092 and +0.02377 for 2324, 2425 and 2526.

### Promoted teams

**Rating a promoted team like the weakest teams sounds right, but predicts worse.**
`promoted_like_bottom` rates a team with no Premier League record like the average of
the worst *n* teams that do have one. Scored only on fixtures involving a promoted
team, over gameweeks 1-6:

| season | promoted | fixtures | bottom 3 | bottom 6 | league average |
|---|---|---|---|---|---|
| 2425 | IPS, LEI, SOU | 17 | **-2.80676** | -2.81500 | -2.83265 |
| 2526 | LEE, SUN | 12 | -2.94684 | -2.91454 | **-2.85005** |
| pooled | | 29 | -2.865 | -2.856 | **-2.840** |

The two seasons disagree, and pooled together they favour rating promoted teams as
average. Ipswich, Leicester and Southampton were all relegated again; Sunderland
started 25/26 near the top of the table. Twenty-nine fixtures aren't enough to settle
it, and the idea is still plausible, so the option is available but off by default. If
a season's data supports it, use `XGTeamConfig(promoted_like_bottom=3)`.

The setting matters less than it might seem anyway. `prior_matches` shrinks a team
with few matches towards the league average, so once a promoted team has played five
matches this setting makes little difference. It only matters in the first few
gameweeks, which is also where there is least data to measure it with.

**The real problem with promoted teams is somewhere else.** The worry behind
`promoted_like_bottom` isn't really the log probability of the scorelines. It's that
promoted teams' players are cheap, so if their team is rated as average, the initial
squad selection might fill up on them. That's a question about squads, so it should be
measured on squads.

It turns out this doesn't happen, for an unexpected reason. Picking a gameweek 1
squad for 2526 under either setting selects **no promoted-team players at all**,
because every one of them is predicted zero points:

| group | players | predicted above zero | best |
|---|---|---|---|
| promoted (LEE, SUN) | 72 | 0 (0.0%) | 0.00 pts |
| everyone else | 618 | 274 (44.3%) | 7.68 pts |

At gameweek 1 there are no current-season matches, so `get_recent_minutes_for_player`
falls back to `estimate_minutes_from_prev_season`. That looks up the player's minutes
in the previous *Premier League* season for their current team. Promoted players have
no such history, so it returns `[0]`: they are predicted zero minutes, and so zero
points. The same happens to every summer signing, which is part of why only 44% of the
other players are predicted above zero.

So the actual bias is the opposite of the worry. At gameweek 1 AIrsenal can't pick too
many promoted-team players, because it can't pick any of them; Sunderland's strong
start to 25/26 was invisible to it. Fixing this means deciding how many minutes to
assume for a player with no history (for example, typical minutes for their position
at their team, the FPL API's `chance_of_playing`, or using their price as a sign of
whether they were signed to start), and that belongs in the minutes model. Until that's
done, `promoted_like_bottom` makes no difference at gameweek 1 either way.

`tools/team_ratings.py` prints the fitted ratings. That's how the two promoted teams
were found to have net ratings of 0.96 and 0.89, almost exactly average, as feared.

### A negative binomial: rejected

`PoissonScorelines` treats the predicted mean as the exact scoring rate for the match.
An obvious alternative is to let the rate itself vary around the mean with a gamma
distribution, which gives a negative binomial distribution over goals. It has the same
mean but fatter tails, so a 5-0 becomes unlikely rather than almost impossible.

This was built, measured and rejected. The held-out log probability was *identical* to
the Poisson's in all three seasons, because the fitted gamma shape parameter hit its
upper limit every time: there was no extra variation in the goal counts for it to
explain. In fact, compared with the model's own predictions, goal counts vary *less*
than a Poisson would:

| season | sides | mean(m) | mean((goals - m)^2) | ratio |
|---|---|---|---|---|
| 2324 | 740 | 1.5750 | 1.4502 | 0.921 |
| 2425 | 1500 | 1.4867 | 1.4004 | 0.942 |
| 2526 | 2260 | 1.4514 | 1.3419 | 0.925 |

A ratio of one would match a Poisson exactly. A gamma mixture can only *add* variance,
and the observed variance is already 6-8% below the mean, so it has nothing to do. What
is needed is a distribution that can be narrower than a Poisson, which is the
Conway-Maxwell-Poisson below.

### Changes to the goal distribution and the ratings

Four ideas were tested here. Since goals vary less than a Poisson predicts, the first
was a distribution that allows that. The other three brought in actual goals or a
second model: adding actual goals to the fitting target alongside expected goals,
averaging the predictions of the `xg` and `extended` models, and giving each team its
own home advantage. Only the first was kept.

All scores in this section are held-out log probability per side of a match (so two
observations per fixture), over gameweeks 5-38 of 2324, 2425 and 2526: 1021 fixtures
and 2042 observations. Where a standard error or t statistic is quoted, it is for the
difference between two versions of the model. Both versions are scored on the same
matches, so the difference can be computed match by match (a paired comparison).

#### Conway-Maxwell-Poisson: kept

A Poisson distribution assumes the variance of a team's goals equals the mean.
Conway-Maxwell-Poisson drops that assumption: the probability of `n` goals is
proportional to `rate ** n / factorial(n) ** dispersion`, so a dispersion of one is a
Poisson and above one is narrower. `ConwayMaxwellScorelines` implements it, but takes
the *mean* rather than the rate as its input, so changing the dispersion changes the
shape of the distribution without changing the predicted number of goals.

The dispersion is chosen by sweeping over values, not fitted along with the rest of the
model. Fitted by maximum likelihood on the same data as the ratings, it comes out at
1.20, but that's biased: the ratings have already been fitted to those matches, so the
leftover variation looks smaller than it really is. On held-out data, each season's
best value is above one, but they disagree about how far above:

| dispersion | 2324 | 2425 | 2526 | pooled |
|---|---|---|---|---|
| 1.00 (Poisson) | -1.54446 | -1.48409 | -1.43324 | -1.48732 |
| 1.10 | -1.54365 | -1.48260 | -1.42933 | -1.48525 |
| 1.15 | -1.54395 | -1.48250 | -1.42798 | -1.48487 |
| **1.17** | **-1.54420** | **-1.48257** | **-1.42755** | **-1.48483** |
| 1.20 | -1.54470 | -1.48280 | -1.42702 | -1.48490 |
| own optimum | 1.09 | 1.14 | 1.31 | **1.17** |

`DEFAULT_GOAL_DISPERSION = 1.17` is the best value across all three seasons, and it's
also better than a Poisson in each season on its own (+0.00026, +0.00152, +0.00569). It's
rounded to two decimal places rather than the 1.171 the sweep gives, because the scores
barely change nearby: everything from 1.15 to 1.20 is within 0.0001 of the best, and
the three seasons' own best values are up to 0.2 apart.

The improvement of +0.0025 over a Poisson is measured on the same three seasons that
were used to choose the dispersion, so it's optimistic. A fairer estimate comes from
leave-one-season-out validation: choose the dispersion on two seasons and score it on
the third. That gives **+0.0017** per side. This is a similar size to the gain from
time weighting, and it's consistent: all three splits choose a dispersion above one
(1.11, 1.19, 1.22).

Its main effect on predictions is to reduce the chance of clean sheets. Against a team
expected to score 1.45 goals, P(clean sheet) drops from 0.2346 to 0.2151 (about 8% in
relative terms), and by more against stronger attacks. `tools/tune_goal_dispersion.py`
recalculates the whole table.

That feeds through to predicted points, which is what ultimately matters.
`backtest_points` over gameweeks 5-30, changing only the scoreline wrapper:

| season | team model | MAE | appeared | RMSE | rank |
|---|---|---|---|---|---|
| 2425 | `extended` | 0.924965 | 1.956485 | 1.909224 | 0.774402 |
| 2425 | xg + Poisson | 0.919409 | 1.945332 | 1.899200 | 0.776003 |
| 2425 | xg + Conway-Maxwell | **0.915531** | **1.938111** | **1.898669** | **0.776307** |
| 2526 | xg + Poisson | 0.903372 | 2.087629 | 1.881445 | 0.816007 |
| 2526 | xg + Conway-Maxwell | **0.899828** | **2.080546** | **1.880465** | **0.816175** |

All four measures improve in both seasons, which is more than the log probability
suggested. The 2425 rows also show `xg` beating `extended` on points, not just on
scorelines.

**This is why `DEFAULT_TEAM_MODEL` is `xg`.** The one downside is that `XGTeamModel`
can't be fitted without expected goals, and the FPL API has only recorded them since
2223. A replay or backtest of an earlier season needs `--team-model extended`, and the
error message says so.

#### Actual goals in the fitting target: rejected

This tried fitting the ratings to a mix of expected and actual goals, controlled by a
`goals_weight` (no longer in the code). The idea was that a team rated on both would
effectively learn how well it converts chances. Every amount of actual goals made the
model worse, and more made it worse still:

| goals weight | pooled gain vs xG alone | t |
|---|---|---|
| 0.25 | -0.00023 | -0.26 |
| 0.50 | -0.00226 | -1.27 |
| 1.00 | -0.01178 | -3.27 |

Two other measurements explain why. Across the league, there's no conversion rate to
correct for: over the three seasons the model predicts 1.5103 goals per side and 1.5064
were scored, a ratio of 0.9974. And for individual teams, it doesn't persist. Splitting
each season at gameweek 19 and correlating each team's goals minus xG in the first half
with the second half:

| season | finishing (for) | keeping (against) |
|---|---|---|
| 2324 | -0.218 | +0.233 |
| 2425 | -0.185 | -0.422 |
| 2526 | +0.107 | -0.478 |

Six correlations, with no consistent sign and a mean of about -0.16. A team that scored
more than its expected goals is, if anything, slightly *less* likely to do so again. A
conversion rate would just be fitted to noise.

#### Averaging the `xg` and `extended` models: rejected

Instead of mixing the fitting targets, this mixed the two models' predicted
distributions: `w * xg + (1 - w) * extended`. It was worth testing separately because
`extended` has a different form and is fitted to goals by MCMC, and two models' errors
can partly cancel out even when one model is worse.

| w (weight on xg) | 2324 | 2425 | 2526 | pooled |
|---|---|---|---|---|
| 0.0 (`extended` alone) | -1.55972 | -1.50251 | -1.44491 | -1.50244 |
| 0.5 | -1.54253 | -1.48859 | -1.43321 | -1.48816 |
| 0.7 | **-1.54067** | -1.48518 | -1.43019 | -1.48540 |
| 0.9 | -1.54206 | -1.48309 | -1.42817 | **-1.48449** |
| 1.0 (`xg` alone) | -1.54420 | **-1.48257** | **-1.42755** | -1.48483 |

The best pooled mixture is 90% `xg` and 10% `extended`, which is +0.00034 better than
`xg` alone (t = 0.64). But that comes from one season, 2324; 2425 and 2526 both prefer
`xg` alone. With leave-one-season-out validation (choosing `w` on two seasons and
scoring the third), the mixture is **worse**: -0.00076. The two models' predicted means
have a correlation of 0.850 and differ by 0.197 goals on average, so `extended` mostly
adds the same information as `xg` plus its own errors, and it's the worse model by
0.018. There's nothing for a mixture to gain.

This was measured by combining both models' saved predictions; a combined model was
never built.

#### A separate home advantage for each team: rejected

`home_mean` and `away_mean` give every team the same home advantage. This tried giving
each team its own multiplier. For the home team it multiplies the goals they're
expected to score and divides the goals they're expected to concede. It's found by
solving `created * x - conceded / x = 0` for the positive root, shrunk towards no
advantage. It looked slightly positive, but isn't:

| prior matches at no advantage | pooled gain | t | 2324 | 2425 | 2526 |
|---|---|---|---|---|---|
| 5 | +0.00116 | +0.76 | +0.00304 | +0.00396 | -0.00353 |
| 20 | +0.00102 | +1.51 | +0.00206 | +0.00255 | -0.00156 |
| 50 | +0.00059 | +1.80 | +0.00109 | +0.00141 | -0.00073 |
| 150 | +0.00024 | +1.97 | +0.00042 | +0.00056 | -0.00026 |

The t statistic *increases* as the effect is shrunk towards zero, while the effect
itself falls to +0.0002, and 2526 goes the other way at every level. Two seasons for and
one against, with an effect that disappears under any shrinkage, isn't a real finding,
and isn't worth an extra rating per team and a quadratic equation to solve.

### Fitting details

#### Number of iterations

The alternating fit stops when no rating changes by more than `tolerance = 1e-12` in a
pass, up to a maximum of `max_iterations = 100` passes. Fitting 2627 gameweek 3 (1160
matches) for `n` passes and comparing with the fully converged result (2000 passes):

| passes | largest step | gap to the fixed point |
|---|---|---|
| 1 | - | 2.5e-02 |
| 2 | 2.4e-02 | 1.1e-03 |
| 3 | 1.0e-03 | 3.1e-05 |
| 5 | 9.0e-07 | 3.7e-08 |
| 10 | 2.6e-12 | 4.6e-15 |
| 20 | 2.2e-16 | 0 |

The error shrinks by a factor of roughly 30 each pass. The same holds in every data
window tested, including the smallest (2324 gameweek 5, 39 matches: 5e-11 after ten
passes), and with the time weighting or the shrinkage prior turned off. Held-out scores
over 2526 gameweeks 5-38 barely depend on the number of passes:

| passes | avg log prob per fixture |
|---|---|
| 1 | -2.85932412 |
| 2 | -2.85925965 |
| 3 | -2.85925779 |
| 5, 10, 50 | -2.85925773 |

These numbers come from `backtest_team_model`, which scores per fixture (both teams'
goals together), whereas the tables above are per side; halve them to compare. Stopping
early costs at most 3.3e-5 per side, a fiftieth of the dispersion improvement. The fit
uses a tolerance with a cap, rather than a fixed number of passes, because how quickly
it converges depends on how well the fixtures connect the teams. In an artificial
four-team schedule where two teams only ever play each other, it's still 1.6e-3 away
after ten passes. Real data converges in seven to thirteen passes.

#### Match weights sum to the number of matches

`XGTeamModel._weights` rescales the time weights `exp(-epsilon * time_diff)` to
`n * weights / weights.sum()`, as bpl's two Dixon-Coles models and
`scale_goals_by_minutes` do, so the weights always sum to the number of matches,
whatever the time weighting. Without rescaling there are two problems.

**The fit would depend on which gameweek you're predicting.** `time_diff` is measured
back from the gameweek being predicted, so predicting further ahead multiplies every
weight by the same constant, while `prior_matches` stays fixed. On the same data (2627,
1160 matches), unrescaled weights sum to 496 when predicting gameweek 38 but 1160 when
predicting gameweek 3, so the prior counts for more and every rating moves towards the
league average: Arsenal's defence is 0.595 rather than 0.573. `airsenal run` fits at
`min(request.gameweeks)`, a few days after the last result, so this shows up most in
`tools/team_ratings.py --gameweek 38`.

**Tuning `epsilon` would also change the amount of shrinkage.** Without rescaling, the
total weight falls as epsilon increases (496 out of a possible 1160 at 0.6), so
`prior_matches = 5` is equivalent to about 11.6 matches for this data, and more at
larger epsilon. With rescaling, the two can be tuned separately (held-out log
probability per side, gameweeks 5-38 of three seasons, 1021 fixtures):

| epsilon | prior 2 | prior 5 | prior 10 | prior 20 | prior 40 |
|---|---|---|---|---|---|
| 0.0 | -1.48469 | -1.48546 | -1.48820 | -1.49392 | -1.50284 |
| 0.4 | -1.48415 | -1.48480 | -1.48731 | -1.49277 | -1.50157 |
| 0.6 | **-1.48411** | -1.48471 | -1.48711 | -1.49242 | -1.50111 |
| 0.9 | -1.48429 | -1.48482 | -1.48706 | -1.49214 | -1.50062 |
| 1.2 | -1.48473 | -1.48519 | -1.48727 | -1.49210 | -1.50033 |
| 1.8 | -1.48613 | -1.48642 | -1.48817 | -1.49250 | -1.50018 |

`epsilon = 0.6` is the best pooled value for both of the two smallest priors.
`prior_matches = 2` looks better than 5 (+0.00060 pooled), but isn't. The seasons
disagree about shrinkage more than about anything else tested here: 2324 is best at 2,
2425 at 10 and 2526 at 20. Choosing the prior on two seasons and scoring the third is
**0.00158 worse**, three times the apparent in-sample gain, so both defaults are
unchanged.

The rescaling itself makes no measurable difference to held-out scores. A backtest fits
at the same gameweek it predicts, so the first problem above can't show up in one:

| weights | 2324 | 2425 | 2526 | pooled |
|---|---|---|---|---|
| unrescaled | -1.54420 | -1.48257 | -1.42755 | -1.48483 |
| largest counts as one | -1.54410 | -1.48258 | -1.42756 | -1.48481 |
| sum to the match count | -1.54347 | -1.48212 | -1.42836 | **-1.48471** |

The dispersion sweep above used unrescaled weights. Repeated with rescaled weights,
1.17 is still the best pooled value, and still better than a Poisson in every season
(+0.00033, +0.00156, +0.00561).

What rescaling does change is the ratings: they are about 12% more spread out, because
`prior_matches = 5` now means five matches rather than 11.6. So a team with few
matches ends up further from the league average. Arsenal's attack and defence are
1.252/0.547 with rescaling and 1.226/0.573 without, and a promoted team with two
matches played gets 29% of the weight on its own record rather than 15%.

## The xG player model

`XGTeamModel` predicts how many goals a team will score from the chances it creates.
`XGPlayerModel` in `player_models/xg.py` predicts which players those goals will go to,
based on the chances each player has had. It uses the same Dirichlet update, pooled
prior, minutes scaling and time weighting as `ConjugatePlayerModel`, but counts
something different. The conjugate model counts the fraction of their team's goals a
player scored. The xG model counts the fraction of their team's *expected* goals the
player was expected to score. It also mixes in a small amount of what the player
actually scored and assisted (see
[Mixing actual goals back in](#mixing-actual-goals-back-in)), because that measures
better than leaving it out. That is the one place it differs from the xG team model.

**It is `DEFAULT_PLAYER_MODEL`.** The rest of this section explains why, and what was
needed to make it work: expected assists have to be scaled to match the assists FPL
awards, and the amount of shrinkage has to differ by position by a factor of about 100.

It gets expected goals and assists from `expected_goals`, `expected_assists` and
`team_expected_goals`, which are `NotRequired` keys on `PlayerFitData`: the values for
each (player, match), and the team total that the player's share is a share of. They
are filled in by `process_player_data` from `get_expected_goals_by_fixture`, the same
query the team model uses.

### Why it should work

These are the same two checks that supported the team model, applied to players. Each
season is split at gameweek 19, using players with over 450 minutes in both halves,
with rates per 90 minutes. The table shows the correlation between each first-half
measure and second-half actual goals or assists:

| season | players | goals -> goals | xG -> goals | assists -> assists | xA -> assists |
|---|---|---|---|---|---|
| 2324 | 250 | 0.644 | **0.725** | 0.495 | **0.533** |
| 2425 | 253 | 0.676 | **0.696** | 0.505 | **0.581** |
| 2526 | 252 | 0.554 | **0.706** | 0.373 | **0.478** |

In every season, for both goals and assists, a player's expected numbers in the first
half of the season predict their actual numbers in the second half better than their
actual first-half numbers do.

Fitting to expected goals throws away how much a player out- or under-performed them,
and correlating that between the two halves shows how much information is lost.
For finishing (goals minus xG per 90) the correlations are +0.091, +0.074 and -0.114:
no consistent sign, the same result as for teams. For chance creation (assists minus
xA per 90) they are +0.255, +0.106 and +0.076: small, but always positive. This is the
one result here that argues for keeping some of the actual numbers, and `goal_weight`
(below) is where that is tested.

There's also simply more data. A team fails to score in between a fifth and a quarter
of its matches (157, 178 and 194 of the 760 team-matches per season), and
`scale_goals_by_minutes` drops those matches when fitting to goals, because a share of
zero goals says nothing about any player. Every match has some expected goals, so
every match counts.

### Expected assists have to be corrected

Expected goals don't need correcting: across the league, teams score about as many
goals as expected, which is the same result `XGTeamModel` found.

| season | goals | xG | goals/xG | assists | xA | assists/xA |
|---|---|---|---|---|---|---|
| 2324 | 1196 | 1199.1 | 0.997 | 1071 | 752.2 | **1.424** |
| 2425 | 1076 | 1093.5 | 0.984 | 971 | 705.6 | **1.376** |
| 2526 | 1005 | 1068.3 | 0.941 | 942 | 683.2 | **1.379** |

Assists are different: FPL awards about 40% more assists than xA credits, every season.
FPL has its own rules for assists, such as for a shot that rebounds in or a pass to a
player who wins a penalty, and xA doesn't count a chance-creating pass in those cases.
Fitted directly to xA, the model would under-predict assists by about 30%.

So `calibrate` scales both columns by how many actual goals and assists the players in
the fitting data got per expected goal and assist. It's calculated per position,
because the model is fitted separately for each position and the positions really do
differ. At 2526 gameweek 20 the factors were:

| | GK | DEF | MID | FWD |
|---|---|---|---|---|
| expected goals -> goals | 0.00 | 0.87 | 1.03 | 0.99 |
| expected assists -> assists | 4.73 | 1.24 | 1.39 | 2.27 |

A forward's FPL assists are about twice their xA, compared with 1.4 times for a
midfielder: the extra FPL-only assists mostly go to players who are close to the ball
when a goal is scored, rather than to whoever made the pass. Goalkeepers score no
goals, so their factor is zero (not a division by zero), which matches what the
conjugate model gives them.

### Each position needs a different amount of shrinkage

`n_goals_prior` sets how many goals' worth of the average for the player's position
a player is credited with before their own record counts. It is the one
hyperparameter here that makes a real difference. A single value for every position is
the wrong approach. Held-out mean log probability by position, gameweeks 5-30 of the
three seasons, predicting one gameweek ahead, with the best value in each row in bold:

| n_goals_prior | 2 | 6 | 10 | 15 | 25 | 60 | 100 | 250 | 700 |
|---|---|---|---|---|---|---|---|---|---|
| GK | -0.07414 | -0.07223 | -0.07130 | -0.07056 | -0.06963 | -0.06825 | -0.06765 | -0.06705 | **-0.06687** |
| DEF | -0.43887 | -0.43536 | -0.43448 | **-0.43424** | -0.43455 | -0.43645 | -0.43806 | -0.44100 | -0.44338 |
| MID | -0.76452 | **-0.76241** | -0.76294 | -0.76420 | -0.76681 | -0.77338 | -0.77781 | -0.78520 | -0.79089 |
| FWD | -0.98311 | -0.97356 | -0.97009 | -0.96796 | -0.96612 | **-0.96493** | -0.96509 | -0.96616 | -0.96742 |

The best values differ by a factor of about 100, and the reason is the same for each
position: how much information a player's own record contains. A midfielder is
involved in enough of their team's goals to be distinguished from the rest of their
position, a forward less reliably, and a goalkeeper almost never. So the best
prediction for a goalkeeper is the same as for every other goalkeeper. Above about 400
the goalkeeper scores stop changing, which is that limit being reached.

So `n_goals_prior` takes one value per position, and the defaults are:

| | GK | DEF | MID | FWD |
|---|---|---|---|---|
| `xg` | 700 | 15 | 6 | 60 |

**This holds up on held-out seasons.** Choosing each position's prior on two seasons
and scoring the third gives -0.62918, compared with -0.62854 if you knew the best
values in advance, and -0.63015 for the best single value (10). So a separate prior per
position is worth +0.00097 over a single value, measured fairly. `ConjugatePlayerModel`
would also benefit from separate priors (-0.63517 held out, against -0.63550 at its
default of 35; its best values are GK 400, DEF 75, MID 25, FWD 150). The gain is a
third as large and doesn't change which model is better. It has been measured but not
implemented.

`PlayerFitData` includes a `position` for this: `process_player_data` is called once
per position, and every player model is fitted separately for each position. If you
build your own training data without a position and ask for a per-position prior, you
get an error rather than, say, a midfielder's shrinkage applied to a goalkeeper.

**Time weighting makes no difference here**, unlike everywhere else in AIrsenal it has
been tested. With all other settings at their final values:

| epsilon | none | 0.0 | 0.1 | 0.2 | 0.4 | 0.6 | 1.2 |
|---|---|---|---|---|---|---|---|
| avg log prob | -0.62844 | -0.62844 | -0.62841 | -0.62839 | **-0.62837** | -0.62839 | -0.62866 |

The whole useful range is worth a hundredth of the per-position prior, and the best
value changes between seasons. `DEFAULT_XG_PLAYER_EPSILON` is 0.2 because that's what
`ConjugatePlayerModel` uses and there's no reason to differ, not because it was chosen
from this table.

### Results

Held-out mean log probability over gameweeks 5-30, one gameweek ahead, all positions,
17984 performances. `xg` uses the defaults (per-position priors, `goal_weight = 0.15`),
and `conjugate` uses a prior of 35, which is both its default and its best single
value:

| season | `xg` | `conjugate` | difference |
|---|---|---|---|
| 2324 | -0.65492 | -0.66549 | +0.01057 |
| 2425 | -0.61563 | -0.62248 | +0.00685 |
| 2526 | -0.61456 | -0.61850 | +0.00394 |
| pooled | **-0.62839** | -0.63551 | +0.00712 |

It's better in every season, and for every position that is involved in goals:

| | GK | DEF | MID | FWD |
|---|---|---|---|---|
| performances | 1196 | 6091 | 8574 | 2123 |
| `conjugate`, prior 35 | -0.06896 | -0.43944 | -0.77116 | -0.96932 |
| `conjugate`, per-position priors | **-0.06649** | -0.43820 | -0.77074 | -0.96656 |
| `xg`, default | -0.06678 | **-0.43390** | **-0.76158** | **-0.96484** |

The middle row is the fairer comparison, because it gives the conjugate model the same
per-position priors (-0.63439 pooled in sample, -0.63517 held out). `xg` is still ahead
by 0.0060 pooled. It's only behind for goalkeepers, by 0.0003 over 1196 performances,
where neither model can predict much.

### Effect on predicted points

On predicted points the two models are much closer. `backtest_breakdown` over the same
gameweeks, for the default `xg` against `conjugate` at its default, with the better of
each pair in bold:

| | 2324 conj | 2324 `xg` | 2425 conj | 2425 `xg` | 2526 conj | 2526 `xg` |
|---|---|---|---|---|---|---|
| points MAE | 0.895196 | **0.891371** | **0.915482** | 0.915723 | **0.900089** | 0.901998 |
| MAE, appeared | 2.018032 | **2.010102** | 1.937966 | **1.937758** | **2.081011** | 2.085379 |
| points RMSE | 1.897641 | **1.891234** | 1.898298 | **1.894697** | 1.880745 | **1.880417** |
| rank correlation | 0.746346 | **0.746605** | 0.776443 | **0.776573** | **0.816141** | 0.816032 |
| attacking MAE | 0.441193 | **0.434552** | 0.451529 | **0.451118** | 0.411261 | **0.411106** |
| involvement MAE, goals | 0.187840 | **0.183846** | 0.174456 | **0.173480** | 0.173716 | **0.172795** |
| involvement MAE, assists | 0.189203 | **0.186902** | **0.178143** | 0.178370 | **0.172905** | 0.173281 |
| performances | 19635 | | 18247 | | 20318 | |

`xg` is better in 15 of the 21 cells. On the three measures that only depend on which
players get the goals (the attacking component and the two involvement errors), `xg`
is better almost everywhere. The exceptions are assists in 2425 and 2526, where the
difference is under 0.0004. `xg` also has the lower RMSE in all three seasons.
`conjugate` has the lower points MAE in two of the three, by 0.0002 and 0.0019; the
next section explains why that measure can mislead. The defending, bonus, cards, saves
and appearance components are identical in every season, which confirms that the only
thing that changed is which players the goals go to.

**A player model can hardly change predicted points at all.** It only affects the
attacking component, which accounts for 0.41 of a total MAE of 0.90. So two models that
agree on most players' shares to two decimal places can't differ by more than a few
thousandths of a point. Points error is the wrong way to compare player models; the log
probability of the shares is the right one, and there `xg` is clearly better.

2425 is the only season with managers in the database, and `score_prediction_breakdown`
skips them because no points model predicts them.

### Why points error rewards under-prediction

If `calibrate` is turned off, the points error *improves* on 2526, by more than any
other change here: MAE 0.891645 against 0.901702, RMSE 1.876950 against 1.880190, and
attacking component MAE 0.397604 against 0.411956 (measured with a single pooled prior
rather than per-position priors). Yet it's the worst model on this page by held-out log
probability: -0.63153 against -0.62839 for the same settings with calibration, and
worse in every season.

The reason is that it predicts fewer goals and assists than actually happen. Summing
each model's predicted goals and assists over every performance where the player
played and their team scored, using the minutes actually played and the goals the team
actually scored (so this only reflects the shares), over all three seasons:

| | goals | of actual | assists | of actual |
|---|---|---|---|---|
| actually happened | 2235 | | 2038 | |
| `conjugate` | 2270 | 1.016 | 2032 | **0.997** |
| `xg`, default | 2289 | 1.024 | 2064 | **1.013** |
| `xg`, `calibrate=False` | 2323 | 1.040 | 1563 | **0.767** |

Without calibration it predicts 23% fewer assists than FPL awards (the gap between xA
and FPL's assists). Most performances score no attacking points, so the prediction that
minimises the error is below the true average, and predicting less of something that
usually doesn't happen lowers both MAE and RMSE. That's why `calibrate` is on by
default, and why the log probability, which can't be improved this way, is the right
measure for the shares.

The same effect, on a smaller scale, is why `n_goals_prior = 2` looks good on
involvement MAE but bad on log probability. Always look at the two together.

### Mixing actual goals back in

`goal_weight` mixes a player's actual goals and assists into the fitting target: zero
means expected goals only, and one gives the same result as the conjugate model. With
the per-position priors above:

| goal_weight | 0.0 | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 | 0.30 | 0.40 | 0.60 |
|---|---|---|---|---|---|---|---|---|---|
| 2324 | -0.65509 | -0.65491 | **-0.65486** | -0.65492 | -0.65510 | -0.65539 | -0.65578 | -0.65689 | -0.66042 |
| 2425 | -0.61661 | -0.61619 | -0.61586 | -0.61563 | -0.61547 | **-0.61540** | -0.61542 | -0.61571 | -0.61734 |
| 2526 | -0.61501 | -0.61477 | -0.61463 | **-0.61456** | -0.61458 | -0.61468 | -0.61486 | -0.61545 | -0.61760 |
| pooled | -0.62891 | -0.62864 | -0.62847 | **-0.62839** | -0.62840 | -0.62851 | -0.62870 | -0.62937 | -0.63181 |

**Unlike for the team model**, where mixing in goals always made things worse, every
season and every position here prefers some actual goals. Defenders, midfielders and
forwards are all best between 0.15 and 0.20, and goalkeepers change by only 0.0002
across the whole range. With leave-one-season-out validation, mixing in actual goals
gains +0.00047 over expected goals alone, and two of the three splits choose 0.15.

So `DEFAULT_XG_GOAL_WEIGHT` is 0.15. What it adds back is what the correlations earlier
suggested: a player's actual goals and assists include penalties and FPL-only assists.
Calibration corrects for those on average, but not for the particular players who
tend to get them. It's worth 0.0005, which is half as much as the per-position prior
and five times as much as time weighting: a real improvement, but a small one.

### Why it's the default

`DEFAULT_PLAYER_MODEL` is `xg` for the same reason `DEFAULT_TEAM_MODEL` is: it's better
on the measure a player model should be judged on in every season and for every
position involved in goals. The margin is 0.0071 pooled, seven times the value of the
per-position prior and fifteen times the value of mixing in goals. On predicted points,
where neither model is measurably better, it's no worse.

Two things to note:

- **The goals-based model is still available.** `--player-model conjugate` selects it,
  and seasons before 2223 need it: like `XGTeamModel`, `XGPlayerModel` raises an error
  if no match has expected goals, rather than silently fitting to something else. A
  default database, which contains the three most recent past seasons, isn't affected.
- **The team model measurements on this page used `conjugate` as the player model.**
  They record what that combination scored, not what a default run scores.

It's still not known whether `xg` picks better squads. Only `airsenal replay` can
answer that, and one run per model isn't enough: the squad optimizer is a genetic
algorithm whose random seed can't be set from the command line, so it needs enough
repeated runs per season to see past the randomness. For now, the case for making it
the default rests on the quality of its shares, which is what a player model predicts.

### Not done

- **A per-position prior for `ConjugatePlayerModel`.** It would help (see
  [above](#each-position-needs-a-different-amount-of-shrinkage)), but by a third as
  much as for `xg`, and it wouldn't change which model is better. `PlayerFitData`
  already includes the position, so it would only take a few lines.
- **A per-position `goal_weight`.** This was measured and isn't worth it: defenders,
  midfielders and forwards are all best between 0.15 and 0.20, and goalkeepers change
  by only 0.0002 across the whole range.
- **`--goal-weight` or `--n-goals-prior` flags.** CLI flags choose which model to use,
  not the settings inside it, which is the rule the `build_*` functions follow. Use
  `XGPlayerConfig` in Python instead.
