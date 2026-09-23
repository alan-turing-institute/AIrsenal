# The xG models: what was measured, and what was rejected

`XGTeamModel` and `XGPlayerModel` are the models AIrsenal predicts with by
default. This is where every number in them came from - each hyperparameter, and
each idea that looked right and measured worse. It is the record the source
comments point at: `DEFAULT_XG_EPSILON`, `DEFAULT_GOAL_DISPERSION`,
`DEFAULT_XG_N_GOALS_PRIOR`, `DEFAULT_XG_GOAL_WEIGHT` and
`promoted_like_bottom` are all decided here rather than in the code that holds
them.

For how to add a model of your own and score it the same way, see
[adding-a-model.md](adding-a-model.md).

## The xG team model

`XGTeamModel` in `team_models/xg.py` predicts a mean and nothing else, and
reaches the points calculation through `ConwayMaxwellScorelines` from
`scorelines.py`, applied in its own table entry. It is an attack-and-defence
rating fitted to the expected goals in past matches rather than to the goals, by
an alternating fit that rates each attack against the defences it actually
faced - which is what separates a good attack from an easy schedule.

Its expected goals arrive as `home_expected_goals` and `away_expected_goals`,
`NotRequired` keys on `TeamFitData` populated by `get_result_dict` from
`get_expected_goals_by_fixture`.

The measurements in this section were taken with `conjugate` as the player
model, and up to the Conway-Maxwell-Poisson section with `PoissonScorelines` as
the wrapper.

**It beats `extended` on every season in the database.** Held-out mean log
probability over gameweeks 5-30, higher being better:

| season | `xg` | `extended` | difference |
|---|---|---|---|
| 2324 | -3.07950 | -3.11516 | +0.03566 |
| 2425 | -2.99039 | -3.02760 | +0.03721 |
| 2526 | -2.85431 | -2.87858 | +0.02428 |

End to end on predicted points, season 2526 GW5-30, it is a small gain
everywhere - and the breakdown says where it comes from:

| metric | `extended` | `xg` |
|---|---|---|
| points MAE | 0.905191 | 0.903423 |
| points MAE, appeared | 2.092136 | 2.088024 |
| points RMSE | 1.883554 | 1.881903 |
| rank correlation | 0.815694 | 0.816195 |
| attacking component MAE | 0.410864 | 0.415394 |
| defending component MAE | 0.272859 | 0.265226 |

**The whole gain is in defending, and attacking is slightly worse.** xG is a
better read on how many a team concedes, and clean sheets and goals-conceded points follow
directly from that, while who takes a team's goals is the player model's job and
has not changed. A better team model on its own reaches predicted points mostly
through the defence.

### Two obvious improvements, both nearly nothing

**Time weighting** is `exp(-epsilon * years ago)`, and `--epsilon` reaches it.
Swept over 2425 and 2526 with
`tools/tune_team_time_weighting.py --model xg`:

| epsilon | 0.0 | 0.3 | 0.6 | 0.9 | 1.2 | 1.8 | 2.5 |
|---|---|---|---|---|---|---|---|
| avg log prob | -2.93330 | -2.93234 | **-2.93224** | -2.93285 | -2.93392 | -2.93674 | -2.94025 |

The optimum is worth a thousandth of a nat over no weighting at all, and it is
the default because it is the optimum, not because it matters. Expected goals
are steadier than goals: Dixon-Coles wants 0.9 and gains from it, this barely
notices. Against `extended` the margin is unchanged: +0.03552, +0.04092,
+0.02377 for 2324, 2425, 2526.

**A promoted team is not an average team** - true, and it predicts worse.
`promoted_like_bottom` rates a side with no record like the mean of the worst
*n* that have one. Scored on only the fixtures a promoted team played in, over
gameweeks 1-6:

| season | promoted | fixtures | bottom 3 | bottom 6 | league average |
|---|---|---|---|---|---|
| 2425 | IPS, LEI, SOU | 17 | **-2.80676** | -2.81500 | -2.83265 |
| 2526 | LEE, SUN | 12 | -2.94684 | -2.91454 | **-2.85005** |
| pooled | | 29 | -2.865 | -2.856 | **-2.840** |

The two seasons disagree and the pooled answer favours the assumption being
wrong. Ipswich, Leicester and Southampton were all relegated straight back;
Sunderland started 25/26 near the top of the table. Twenty-nine fixtures cannot
settle it, and the story is still the more plausible one - so the mechanism
ships and the default does not use it. If a season's data supports it,
`XGTeamConfig(promoted_like_bottom=3)` is one argument away.

Worth noting what the shrinkage does here anyway: `prior_matches` pulls a team
with a thin record towards the league average, so by the time a promoted side
has played five matches this setting barely matters. It only bites in the first
few gameweeks, which is exactly where the sample is smallest.

### The gamma has nothing to explain

The obvious use for the gamma that team xG follows is not the rating fit but the
step after it: `PoissonScorelines` treats the predicted mean as the exact rate
for the match, and letting the rate be gamma-distributed about it instead gives
a negative binomial over goal counts - same mean, fatter tails, so a 5-0 is
unlikely rather than incredible.

Built, measured and rejected. The held-out log probability was
*bit-identical* to the Poisson in all three seasons, because the fitted gamma
shape pinned at its ceiling every time: the mixture found no excess dispersion
to explain. Goal counts, given the model's own predictions for them, are if
anything narrower than Poisson:

| season | sides | mean(m) | mean((goals - m)^2) | ratio |
|---|---|---|---|---|
| 2324 | 740 | 1.5750 | 1.4502 | 0.921 |
| 2425 | 1500 | 1.4867 | 1.4004 | 0.942 |
| 2526 | 2260 | 1.4514 | 1.3419 | 0.925 |

A ratio of one is exactly Poisson. A mixture can only *add* variance, so with
the residual scatter already 6-8% below the mean there is nothing for it to do.
The useful direction is the opposite one, a family that can be narrower than
Poisson - which is the Conway-Maxwell-Poisson below.

### Promoted teams: the real problem is elsewhere

The concern behind `promoted_like_bottom` is not the log probability of a
scoreline. It is that promoted-team players are cheap, so if their team is rated
average, the initial squad optimisation should load up on bargains. That is a
question about squads, so it wants measuring on squads.

It does not happen, and the reason is worth knowing. Building a gameweek 1 squad
for 2526 under both assumptions picks **no promoted-team players at all** -
because every one of them is predicted zero points:

| group | players | predicted above zero | best |
|---|---|---|---|
| promoted (LEE, SUN) | 72 | 0 (0.0%) | 0.00 pts |
| everyone else | 618 | 274 (44.3%) | 7.68 pts |

At gameweek 1 there are no current-season matches, so
`get_recent_minutes_for_player` falls back to
`estimate_minutes_from_prev_season`, which reads the previous *Premier League*
season for the team the player is at now. A promoted-team player has no such
history, so it returns `[0]`, the minutes are zero, and the points model
short-circuits. The same applies to every summer signing, which is part of why
only 44% of the rest are above zero either.

So the bias runs the opposite way to the worry: AIrsenal cannot over-pick
promoted-team players at gameweek 1 because it cannot pick them at all -
Sunderland's start to 25/26 was invisible to it. Fixing that is a decision about
what a player with no history should be assumed to play (their team's typical
minutes for the position, the FPL API's `chance_of_playing`, or their price as a
signal of whether they were bought to start), and it belongs to the minutes
model. Until then, `promoted_like_bottom` cannot matter at gameweek 1 whichever
way it is set.

`tools/team_ratings.py` prints the fitted ratings, which is how the two newly
promoted teams were found sitting at a net 0.96 and 0.89 - almost exactly
average, as the worry supposed.

### Four things tried on the goal distribution and the ratings

Following the gamma finding above: goals are narrower than Poisson given the
model's own means, so the family to try is one that *can* be narrower. Alongside
it, three questions about bringing goals or a second model in - whether goals
belong in the fitting target next to expected goals, whether mixing the two
existing models' predictions beats either, and whether each team should have its
own home advantage. One of the four survived.

Everything below is held-out log probability per side of a match, over gameweeks
5-38 of 2324, 2425 and 2526 - 1021 fixtures, 2042 observations. The paired
standard error is over observations, which is why it can be quoted at all: the
two variants see the same matches, so their difference is paired.

#### Conway-Maxwell-Poisson: kept

The Poisson asserts that a team's goals have a variance equal to their mean.
Conway-Maxwell-Poisson lets that go - the pmf is proportional to
`rate ** n / factorial(n) ** dispersion`, so one is exactly Poisson, above one
is narrower. `ConwayMaxwellScorelines` is it, parameterised by the *mean* rather
than the rate, so that changing the dispersion changes the shape and leaves the
predicted number of goals where it was.

The dispersion is swept, not fitted while the model is. Fitted in sample by
maximum likelihood it comes out at 1.20, and that is biased: the ratings have
already been fitted to the same matches, so the residuals look narrower than
they are. Held out, every season's own optimum is above one but they disagree
about how far:

| dispersion | 2324 | 2425 | 2526 | pooled |
|---|---|---|---|---|
| 1.00 (Poisson) | -1.54446 | -1.48409 | -1.43324 | -1.48732 |
| 1.10 | -1.54365 | -1.48260 | -1.42933 | -1.48525 |
| 1.15 | -1.54395 | -1.48250 | -1.42798 | -1.48487 |
| **1.17** | **-1.54420** | **-1.48257** | **-1.42755** | **-1.48483** |
| 1.20 | -1.54470 | -1.48280 | -1.42702 | -1.48490 |
| own optimum | 1.09 | 1.14 | 1.31 | **1.17** |

`DEFAULT_GOAL_DISPERSION = 1.17` is the pooled held-out optimum, and it is also
better than a Poisson in each season taken alone (+0.00026, +0.00152, +0.00569).
Two decimal places rather than the 1.171 the sweep returns, because the surface
is flat: everything from 1.15 to 1.20 is within a ten-thousandth of a nat of the
peak, and the three seasons put their own optima 0.2 apart.

The gain of +0.0025 against a Poisson is measured at the dispersion that was
chosen on the same three seasons, so the honest figure is the
leave-one-season-out one: choose the pooled optimum on two seasons, score the
third, and it is **+0.0017 nats** a side. That is the same order as the time
weighting, and the direction is unanimous - each of the three
folds picks a dispersion above one (1.11, 1.19, 1.22).

What it does downstream is reduce clean sheets: against a side expected to
create 1.45 goals, P(clean sheet) goes from 0.2346 to 0.2151, about 8% relative,
and more than that against the better attacks. `tools/tune_goal_dispersion.py`
re-derives the whole table.

That reaches the points, which is the number that decides it. `backtest_points`
over gameweeks 5-30, with everything but the scoreline wrapper held fixed:

| season | team model | MAE | appeared | RMSE | rank |
|---|---|---|---|---|---|
| 2425 | `extended` | 0.924965 | 1.956485 | 1.909224 | 0.774402 |
| 2425 | xg + Poisson | 0.919409 | 1.945332 | 1.899200 | 0.776003 |
| 2425 | xg + Conway-Maxwell | **0.915531** | **1.938111** | **1.898669** | **0.776307** |
| 2526 | xg + Poisson | 0.903372 | 2.087629 | 1.881445 | 0.816007 |
| 2526 | xg + Conway-Maxwell | **0.899828** | **2.080546** | **1.880465** | **0.816175** |

All four measures improve in both seasons, which is more than the log
probability promised - and the 2425 rows also show `xg` beating `extended` on
points and not only on scorelines.

**This is why `DEFAULT_TEAM_MODEL` is `xg`.** The one thing it gives up is reach:
`XGTeamModel` cannot be fitted where there are no expected goals to fit it to,
and the FPL API has only recorded them since 2223, so a replay or backtest of an
earlier season has to ask for `--team-model extended`. The error says so.

#### Goals in the fitting target: rejected

Ensembling xG with goals, as a `goals_weight` on the fitting target so that a
team rated on both gets a shrunk conversion factor between them for free. Every
amount of it was worse, monotonically:

| goals weight | pooled gain vs xG alone | t |
|---|---|---|
| 0.25 | -0.00023 | -0.26 |
| 0.50 | -0.00226 | -1.27 |
| 1.00 | -0.01178 | -3.27 |

Two other measurements say why. At league level there is no conversion factor to
fit: over the three seasons the model predicts 1.5103 goals per side and 1.5064
were scored, a ratio of 0.9974. And at team level it does not persist - split
each season at gameweek 19 and correlate a team's goals-over-xG in the first
half against the second:

| season | finishing (for) | keeping (against) |
|---|---|---|
| 2324 | -0.218 | +0.233 |
| 2425 | -0.185 | -0.422 |
| 2526 | +0.107 | -0.478 |

Six measurements, no consistent sign, mean about -0.16. A team that outscored
its expected goals is, if anything, slightly *less* likely to do it next.
Whatever a conversion factor would be fitted to is noise.

#### An ensemble of the two models: rejected

The other reading of "ensemble": not blending the fitting *targets*, but mixing
the two models' predicted distributions, `w * xg + (1 - w) * extended`. Worth
testing separately, because `extended` is a different functional form fitted to
goals by MCMC, and two models' errors can decorrelate even when one is worse.

| w (weight on xg) | 2324 | 2425 | 2526 | pooled |
|---|---|---|---|---|
| 0.0 (`extended` alone) | -1.55972 | -1.50251 | -1.44491 | -1.50244 |
| 0.5 | -1.54253 | -1.48859 | -1.43321 | -1.48816 |
| 0.7 | **-1.54067** | -1.48518 | -1.43019 | -1.48540 |
| 0.9 | -1.54206 | -1.48309 | -1.42817 | **-1.48449** |
| 1.0 (`xg` alone) | -1.54420 | **-1.48257** | **-1.42755** | -1.48483 |

The pooled optimum is a 0.9/0.1 mixture, worth +0.00034 nats over `xg` alone at
t = 0.64 - and it is one season carrying it, 2324, while 2425 and 2526 both
prefer pure `xg`. Held out properly, choosing `w` on two seasons and scoring the
third, the mixture **loses**: -0.00076 nats. The two models' predicted means
correlate at r = 0.850 and differ by 0.197 goals on average, so `extended`
brings mostly the same information as `xg` plus its own error, and it is the
worse model by 0.018 nats. There is nothing for a mixture to recover.

This was measured from both models' stored distributions, not built.

#### A home advantage per team: rejected

`home_mean` and `away_mean` say home is worth the same to everyone. Giving each
team its own multiplier - one number, tilting a match by multiplying what the
home side creates and dividing what it concedes, solved for by the positive root
of `created * x - conceded / x = 0` with shrinkage towards no tilt - looked
mildly positive and is not:

| prior matches at no advantage | pooled gain | t | 2324 | 2425 | 2526 |
|---|---|---|---|---|---|
| 5 | +0.00116 | +0.76 | +0.00304 | +0.00396 | -0.00353 |
| 20 | +0.00102 | +1.51 | +0.00206 | +0.00255 | -0.00156 |
| 50 | +0.00059 | +1.80 | +0.00109 | +0.00141 | -0.00073 |
| 150 | +0.00024 | +1.97 | +0.00042 | +0.00056 | -0.00026 |

The t statistic *rises* as the effect is shrunk towards nothing while the effect
itself collapses to +0.0002 nats, and 2526 disagrees in sign at every level. Two
seasons for and one against, at an effect size that vanishes under any
shrinkage, is not a finding, and not worth an extra rating per team and a
quadratic solve.

#### The alternating fit's iteration count

The fit stops when a pass moves no rating by more than `tolerance = 1e-12`,
capped at `max_iterations = 100` passes. Fitting the same data for `n` passes
and comparing with the fixed point (2000 passes) on 2627 GW3, 1160 matches:

| passes | largest step | gap to the fixed point |
|---|---|---|
| 1 | - | 2.5e-02 |
| 2 | 2.4e-02 | 1.1e-03 |
| 3 | 1.0e-03 | 3.1e-05 |
| 5 | 9.0e-07 | 3.7e-08 |
| 10 | 2.6e-12 | 4.6e-15 |
| 20 | 2.2e-16 | 0 |

So it converges geometrically at roughly a factor of 30 a pass, and the same
holds in every window tried, including the sparsest (2324 GW5, 39 matches: 5e-11
at ten passes) and with the time weighting or the shrinkage prior turned off.
Held-out scores over 2526 GW5-38 barely notice the pass count:

| passes | avg log prob per fixture |
|---|---|
| 1 | -2.85932412 |
| 2 | -2.85925965 |
| 3 | -2.85925779 |
| 5, 10, 50 | -2.85925773 |

These are `backtest_team_model`'s numbers, which are per fixture - both goal
counts - where the tables above are per side; halve them to compare. An
unconverged fit costs 3.3e-5 nats a side at worst, a fiftieth of the dispersion
effect. The count is a tolerance with a cap rather than a fixed number because
how fast the fit converges depends on how well the schedule connects the teams:
a contrived four-team schedule where two teams only ever play each other is
still 1.6e-3 short after ten passes. Real windows settle in seven to thirteen
passes.

#### The weights sum to the number of matches

`XGTeamModel._weights` rescales `exp(-epsilon * time_diff)` to
`n * weights / weights.sum()`, as bpl's two Dixon-Coles models and
`scale_goals_by_minutes` do, so the weights sum to the number of matches
whatever the time weighting. Unnormalised weights go wrong in two ways.

**The fit would depend on how far ahead you aimed it.** `time_diff` is measured
back from the gameweek being predicted, so aiming a season further ahead
multiplies every weight by the same constant while `prior_matches` stays in
absolute units. On identical data (2627, 1160 matches) unnormalised weights sum
to 496 aimed at GW38 against 1160 aimed at GW3, so every rating creeps towards
the league average - ARS defence 0.595 rather than 0.573. `airsenal run` fits at
`min(request.gameweeks)`, days after the last result, so this shows most in
`tools/team_ratings.py --gameweek 38`.

**A sweep over `epsilon` would sweep the shrinkage too.** Unnormalised, total
weight falls as epsilon rises - 496 of a possible 1160 at 0.6 - so
`prior_matches = 5` is worth about 11.6 matches at this window, and more at a
larger epsilon. Rescaled, the two can be chosen separately (held-out log
probability per side, gameweeks 5-38 of three seasons, 1021 fixtures):

| epsilon | prior 2 | prior 5 | prior 10 | prior 20 | prior 40 |
|---|---|---|---|---|---|
| 0.0 | -1.48469 | -1.48546 | -1.48820 | -1.49392 | -1.50284 |
| 0.4 | -1.48415 | -1.48480 | -1.48731 | -1.49277 | -1.50157 |
| 0.6 | **-1.48411** | -1.48471 | -1.48711 | -1.49242 | -1.50111 |
| 0.9 | -1.48429 | -1.48482 | -1.48706 | -1.49214 | -1.50062 |
| 1.2 | -1.48473 | -1.48519 | -1.48727 | -1.49210 | -1.50033 |
| 1.8 | -1.48613 | -1.48642 | -1.48817 | -1.49250 | -1.50018 |

`epsilon = 0.6` is the pooled optimum at both of the two smallest priors.
`prior_matches` looks retunable - 2 is worth +0.00060 pooled over 5 -
and is not. The seasons disagree about shrinkage more than about anything else
tried here: 2324 wants 2, 2425 wants 10, 2526 wants 20. Choosing the prior on
two seasons and scoring the third **loses 0.00158**, three times what choosing
it in sample appears to gain, so both defaults stay where they are.

The rescaling itself is worth nothing measurable on held-out scores - a
backtest fits at the gameweek it then predicts, so the first problem cannot
appear in one:

| weights | 2324 | 2425 | 2526 | pooled |
|---|---|---|---|---|
| unrescaled | -1.54420 | -1.48257 | -1.42755 | -1.48483 |
| largest counts as one | -1.54410 | -1.48258 | -1.42756 | -1.48481 |
| sum to the match count | -1.54347 | -1.48212 | -1.42836 | **-1.48471** |

The dispersion sweep above used unrescaled weights. Re-swept on rescaled ones,
1.17 is still the pooled optimum, and positive in every season on its own
(+0.00033, +0.00156, +0.00561 against a Poisson).

What rescaling does change is the ratings, by about 12% more spread -
`prior_matches = 5` means five matches rather than the window's 11.6 - so a team
with a short record sits further from the league average. ARS reads 1.252/0.547
rescaled against 1.226/0.573 unrescaled, and a promoted side with two matches
played gets 29% of its own record rather than 15%.

## The xG player model

`XGTeamModel` asks how many goals a team will score from the chances it creates.
`XGPlayerModel` in `player_models/xg.py` asks who those goals belong to, from
the chances that player took: it is `ConjugatePlayerModel`'s Dirichlet update,
its pooled prior, its minutes scaling and its time weighting, over a different
count. Where the conjugate model counts what fraction of its team's goals a
player scored, this counts what fraction of its team's *expected* goals the
player was expected to score - not quite purely, because a sixth of what he
actually did measures better than none of it, which is the one place this
departs from the team model's answer.

**It is `DEFAULT_PLAYER_MODEL`.** The rest of this section is why, and what it
took: expected assists have to be calibrated to the assists FPL awards, and the
shrinkage has to vary by position by two orders of magnitude.

Its expected goals and assists arrive as `expected_goals`, `expected_assists`
and `team_expected_goals`, `NotRequired` keys on `PlayerFitData` - the (player,
match) values and the team total that a share of one is a share of - populated
by `process_player_data` from `get_expected_goals_by_fixture`, the same query
the team model uses.

### Why it should work

The same two measurements that justified the team model, asked of players.
Season split at gameweek 19, players with over 450 minutes in both halves, rates
per 90:

| season | players | goals -> goals | xG -> goals | assists -> assists | xA -> assists |
|---|---|---|---|---|---|
| 2324 | 250 | 0.644 | **0.725** | 0.495 | **0.533** |
| 2425 | 253 | 0.676 | **0.696** | 0.505 | **0.581** |
| 2526 | 252 | 0.554 | **0.706** | 0.373 | **0.478** |

What a player was expected to do in the first half of a season predicts what he
actually does in the second half better than what he actually did does, in every
season and for both goals and assists.

What gets thrown away by fitting to the expectation is the over-performance, and
correlating that across the halves says how much is being lost. Finishing -
goals minus xG per 90 - gives +0.091, +0.074, -0.114: no consistent sign, the
same answer team finishing gave. Creating - assists minus xA per 90 - gives
+0.255, +0.106, +0.076, which is small, always positive, and the one thing here
that argues for keeping some of the realised numbers; `goal_weight` below is
where that argument is measured.

There is also simply more of it. A team is goalless in a fifth to a quarter of
its matches - 157, 178 and 194 of the 760 sides a season - and
`scale_goals_by_minutes` drops those from a goals fit, because a share of no
goals is not evidence about anybody. Every match has expected goals in it, so
every match counts.

### Expected assists are not assists, and that has to be corrected

Expected goals need no correction: the league scores what it is expected to,
which is the same fact `XGTeamModel` found at team level.

| season | goals | xG | goals/xG | assists | xA | assists/xA |
|---|---|---|---|---|---|---|
| 2324 | 1196 | 1199.1 | 0.997 | 1071 | 752.2 | **1.424** |
| 2425 | 1076 | 1093.5 | 0.984 | 971 | 705.6 | **1.376** |
| 2526 | 1005 | 1068.3 | 0.941 | 942 | 683.2 | **1.379** |

Assists are a different matter: FPL awards about 40% more of them than xA
credits anyone with, every season. It awards them by its own rules - the shot
that rebounds in, the pass to a player who wins a penalty - and no
chance-creating pass is measured for those. Fitted straight, the model would
under-predict every assist by 30%.

So `calibrate` scales both columns by what the window's own players converted
them into, fitted from the same data as the rest of the model. It is fitted per
position, because that is the unit a player model is fitted on, and the
positions genuinely differ - at 2526 GW20 the factors were:

| | GK | DEF | MID | FWD |
|---|---|---|---|---|
| expected goals -> goals | 0.00 | 0.87 | 1.03 | 0.99 |
| expected assists -> assists | 4.73 | 1.24 | 1.39 | 2.27 |

A forward's assists are twice his xA, where a midfielder's are 1.4 times: the
FPL-only assists land disproportionately on players who are near the ball when
it goes in rather than on the ones who passed it there. Goalkeepers score no
goals at all, which is a zero rather than a division by zero, and the same
answer the conjugate model gives them.

### Every position wants a different amount of shrinkage

`n_goals_prior` is how many goals' worth of the pooled squad a player is
credited with before his own record counts, and it is the one hyperparameter
here that measurably matters. One number for the whole league is the wrong
shape for it: held-out mean log probability by position, gameweeks 5-30 of the
three seasons at horizon 1, the best of each row in bold:

| n_goals_prior | 2 | 6 | 10 | 15 | 25 | 60 | 100 | 250 | 700 |
|---|---|---|---|---|---|---|---|---|---|
| GK | -0.07414 | -0.07223 | -0.07130 | -0.07056 | -0.06963 | -0.06825 | -0.06765 | -0.06705 | **-0.06687** |
| DEF | -0.43887 | -0.43536 | -0.43448 | **-0.43424** | -0.43455 | -0.43645 | -0.43806 | -0.44100 | -0.44338 |
| MID | -0.76452 | **-0.76241** | -0.76294 | -0.76420 | -0.76681 | -0.77338 | -0.77781 | -0.78520 | -0.79089 |
| FWD | -0.98311 | -0.97356 | -0.97009 | -0.96796 | -0.96612 | **-0.96493** | -0.96509 | -0.96616 | -0.96742 |

Two orders of magnitude between the ends of that, and it is the same fact in
four places: how much of a player's own record there is to fit. A midfielder is
involved in enough of his team's goals to be told apart from his squad, a
forward less reliably, and a goalkeeper never - so the best thing to tell the
model about a goalkeeper is that he is like every other goalkeeper. Above about
400 the goalkeeper row stops moving, which is that limit being reached.

So `n_goals_prior` takes one number per position, and the default is

| | GK | DEF | MID | FWD |
|---|---|---|---|---|
| `xg` | 700 | 15 | 6 | 60 |

**Chosen held out, and worth it held out.** Choosing each position's prior on
two seasons and scoring the third gives -0.62918, against -0.62854 for an
oracle that knew the answer and -0.63015 for the best single number (10). So
the per-position prior is worth +0.00097 over one number, honestly measured,
which is more than the gap it had to clear. `ConjugatePlayerModel` gains from
the same idea (-0.63517 held out against -0.63550 at its default 35, wanting
GK 400, DEF 75, MID 25, FWD 150), which is a third as much and does not change
the ordering; it is not implemented there, only measured.

`PlayerFitData` carries a `position` for this: `process_player_data` is
called once per position and every model here is fitted per position. A caller
who assembles their own training data and asks for a per-position prior is told
what is missing rather than given a midfielder's shrinkage for a goalkeeper.

**Time weighting is worth nothing here**, unlike everywhere else in this
package that has been swept for it. At the final configuration:

| epsilon | none | 0.0 | 0.1 | 0.2 | 0.4 | 0.6 | 1.2 |
|---|---|---|---|---|---|---|---|
| avg log prob | -0.62844 | -0.62844 | -0.62841 | -0.62839 | **-0.62837** | -0.62839 | -0.62866 |

The whole useful range is a hundredth of what the prior is worth, and the
nominal optimum moves between seasons. `DEFAULT_XG_PLAYER_EPSILON` is 0.2
because that is what `ConjugatePlayerModel` uses and there is no reason to
differ, not because it was chosen on this table.

### What it scores

Held-out mean log probability over gameweeks 5-30, horizon 1, every position,
17984 performances. `xg` is the shipped configuration - per-position priors,
`goal_weight = 0.15` - and `conjugate` is at 35, which is its default and its
own pooled optimum:

| season | `xg` | `conjugate` | difference |
|---|---|---|---|
| 2324 | -0.65492 | -0.66549 | +0.01057 |
| 2425 | -0.61563 | -0.62248 | +0.00685 |
| 2526 | -0.61456 | -0.61850 | +0.00394 |
| pooled | **-0.62839** | -0.63551 | +0.00712 |

Better in every season, and better at every position that touches a goal:

| | GK | DEF | MID | FWD |
|---|---|---|---|---|
| performances | 1196 | 6091 | 8574 | 2123 |
| `conjugate`, prior 35 | -0.06896 | -0.43944 | -0.77116 | -0.96932 |
| `conjugate`, per-position priors | **-0.06649** | -0.43820 | -0.77074 | -0.96656 |
| `xg`, shipped | -0.06678 | **-0.43390** | **-0.76158** | **-0.96484** |

The middle row is the fair comparison rather than the flattering one: the same
per-position idea helps the goals model too, by a third as much (-0.63439
pooled, in sample; -0.63517 held out), and it is measured here rather than
implemented there. `xg` is ahead of it by 0.0060 pooled, and behind it only for
goalkeepers, by 0.0003 over 1196 performances - a position where neither model
has anything to say and both are saying it.

### End to end on points, where it is much closer

`backtest_breakdown` over the same gameweeks, the shipped `xg` against
`conjugate` at its default. The better of each pair is in bold:

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

`xg` wins fifteen of the twenty-one cells. The three
measures that are only about who the goals belong to - the attacking component
and the two involvement errors, apart from assists in the two seasons where they
differ by 0.0004 - go to `xg` almost everywhere, and RMSE does in all three
seasons. Points MAE goes to `conjugate` in two of the three, by 0.0002 and
0.0019, which is the measure the section below is about. The defending, bonus,
cards, saves and appearance components are identical to the last digit in every
season, which is the check that the only thing that changed is who the goals
belong to.

**A player model can barely move predicted points at all**, and that is worth
saying plainly. It reaches a score through the attacking component alone - 0.41
of a total MAE of 0.90 - so two models that agree about most players to two
decimal places of a share cannot differ by more than a few thousandths of a
point. This is the wrong instrument for the question; the log probability of the
shares is the right one, and it is not close.

2425 is the one season with managers in the database, and
`score_prediction_breakdown` skips them: no points model predicts the position.

### The points error rewards under-prediction, and the calibration proves it

Turn `calibrate` off and the points error *improves*, on 2526, by more than
anything else here moves it: MAE 0.891645 against 0.901702, RMSE 1.876950
against 1.880190, attacking component MAE 0.397604 against 0.411956 (measured
at a single pooled prior rather than the per-position ones). It is also the worst model in the
file by held-out log probability - -0.63153 against -0.62839 for the same
configuration calibrated, and worse in every season.

The reason is that it predicts fewer attacking returns than happen. Summing each
model's predicted goals and assists over every performance where the player
appeared and their team scored - conditioned on the minutes actually played and
the goals the team actually scored, so this is the level of the shares alone -
over all three seasons:

| | goals | of actual | assists | of actual |
|---|---|---|---|---|
| actually happened | 2235 | | 2038 | |
| `conjugate` | 2270 | 1.016 | 2032 | **0.997** |
| `xg`, shipped | 2289 | 1.024 | 2064 | **1.013** |
| `xg`, `calibrate=False` | 2323 | 1.040 | 1563 | **0.767** |

Uncalibrated, it predicts 23% fewer assists than are awarded - the gap between
xA and FPL's assists - and predicting fewer of a thing that mostly does not
happen lowers the mean error against it. Most performances score no
attacking points, so the error-minimising prediction is below the mean, and both
MAE and RMSE pay for the bias in the direction the truth is not. That is why
`calibrate` defaults to on and why the log probability, which cannot be gamed
this way, is the number to read for the shares.

The same effect, smaller, is what makes `n_goals_prior = 2` look good on the
involvement MAE and terrible on the log probability. Read the two together.

### Blending the real goals back in, which the team model could not use

`goal_weight` mixes the realised involvement into the fitting target, so zero is
pure expected goals and one is the conjugate model reached the long way round.
At the per-position priors above:

| goal_weight | 0.0 | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 | 0.30 | 0.40 | 0.60 |
|---|---|---|---|---|---|---|---|---|---|
| 2324 | -0.65509 | -0.65491 | **-0.65486** | -0.65492 | -0.65510 | -0.65539 | -0.65578 | -0.65689 | -0.66042 |
| 2425 | -0.61661 | -0.61619 | -0.61586 | -0.61563 | -0.61547 | **-0.61540** | -0.61542 | -0.61571 | -0.61734 |
| 2526 | -0.61501 | -0.61477 | -0.61463 | **-0.61456** | -0.61458 | -0.61468 | -0.61486 | -0.61545 | -0.61760 |
| pooled | -0.62891 | -0.62864 | -0.62847 | **-0.62839** | -0.62840 | -0.62851 | -0.62870 | -0.62937 | -0.63181 |

**Unlike the same idea at team level**, where blending goals into an xG fit was
monotonically worse, every season prefers a blend and every position does too -
DEF, MID and FWD all optimise between 0.15 and 0.20, and the goalkeepers move by
0.0002 across the whole range and have no opinion. Held out, choosing the weight
on two seasons and scoring the third gains +0.00047 over no blend, and two of
the three splits choose 0.15.

`DEFAULT_XG_GOAL_WEIGHT` is therefore 0.15. What it is recovering is what the
persistence measurements said would be there: a player's realised goals carry
penalties and the FPL-only assists, which calibration corrects for on average
but not for the player who actually takes them. It is worth 0.0005, which is
half of what the per-position prior is worth and five times what time weighting
is - real, small, and measured the same way as the rest.

### It is the default player model

`DEFAULT_PLAYER_MODEL` is `xg`, on the same standard `xg` became
`DEFAULT_TEAM_MODEL`: it wins the measure a player model is scored on in every
season and at every position that touches a goal, by 0.0071 pooled - seven times
what the per-position prior is worth and fifteen times the blend - and it does
not lose end to end, where nothing measurably wins.

Two things follow:

- **The goals-fitted model has not gone anywhere.** `--player-model conjugate`
  selects it, and a season before 2223 needs it: `XGPlayerModel` refuses to fit
  where no match has expected goals rather than quietly fitting to something
  else, exactly as `XGTeamModel` does. A default database - three past seasons -
  is unaffected.
- **The team model's measurements were taken with `conjugate` as the player
  model.** They are the record of what that configuration scored, not a claim
  about what a default run scores.

What is still not settled is whether it picks better squads. `airsenal replay`
is the only measure that answers that, and one run per model does not: the
squad optimizer is a genetic algorithm whose seed no flag exposes, so it needs
enough repeats per season to see past its own randomness. The case for the
default here is the shares, which is what the model is.

### What was not done

- **A per-position prior for `ConjugatePlayerModel`.** It gains from one -
  -0.63517 held out against -0.63550 at its default, wanting GK 400, DEF 75,
  MID 25, FWD 150 - which is a third of what `xg` gains and does not change the
  ordering between them. `PlayerFitData` carries the position, so it is a few
  lines whenever it is wanted.
- **A per-position `goal_weight`.** Measured, and there is nothing there:
  defenders, midfielders and forwards all optimise between 0.15 and 0.20, and
  goalkeepers move by 0.0002 across the whole range.
- **A `--goal-weight` or `--n-goals-prior` flag.** The CLI takes the flags that
  name a model, not the knobs inside one, which is the rule `build_*` functions
  follow. `XGPlayerConfig` is one import away in Python.
