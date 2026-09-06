# Plan: what a prediction model is allowed to be

The prediction protocols assume more about a model than they need to. A model
must be probabilistic, it must predict one particular quantity, and the way its
prediction is turned into points is fixed. This is the plan for moving those
seams, in phases that can be shipped and scored one at a time.

Status: phases 0 to 4 are done. Phase 5 is all that is left, and it is gated on
having a consumer for the uncertainty it would carry.

## What is actually assumed

Five separate assumptions, only two of which are about probability.

1. **The player model's output quantity is fixed, not just its form.**
   `PlayerModel.get_probs` (`prediction/protocols.py`) demands
   `prob_score`/`prob_assist`/`prob_neither` - a share of *team goals*, as three
   point estimates. A model whose native output is per-player xG/xA, or expected
   points directly, has to launder itself through a multinomial share, and a
   probabilistic model must collapse its posterior to a mean before
   `prediction/points.py` sees it. `NumpyroPlayerModel`'s uncertainty is
   discarded at `get_probs`.
2. **`get_probs` returns `dict[str, np.ndarray]` with its keys described in
   prose.** This is the same defect `680862d3` fixed on the *input* side by
   introducing `PlayerFitData` and `TeamFitData`. The migration was never
   finished on the output side.
3. **`predict_score_n_proba` requires a discrete pmf over goal counts**, and
   `points.py` consumes only the two marginals independently - attacking points
   from the score distribution, clean sheets from the concede distribution. A
   model that predicts a continuous expected-goals quantity has to fabricate a
   pmf; a model that predicts a correlated scoreline has the correlation thrown
   away.
4. **The points composition is hard-wired.** `calc_predicted_points_for_player`
   takes seven positional pandas objects and sums six bespoke free functions.
   Bonus, cards, saves and defensive contributions have no protocol, no table
   and no flag - `PointsConfig` only turns them off. `docs/adding-a-model.md`
   says five things are pluggable; these four are not. By CLAUDE.md's own test
   ("if adding one seems to need edits anywhere but the class and its table
   line, the seam is in the wrong place"), adding a points model today is a
   rewrite of `points.py`.
5. **The minutes "weighted average" is an unweighted one.** `points.py`
   evaluates expected points at each of the last three minutes values and
   divides by three - a three-sample empirical minutes distribution with uniform
   weights, hidden inside the points calculation, which also forces every
   component to be a function of one scalar minutes value.

Downstream, it all collapses to a single `predicted_points` float
(`db/models.py`), and `prediction/evaluation.py` scores log-probability only -
so a non-probabilistic model cannot be scored by anything short of a full
`airsenal replay`.

## Decisions already recorded that constrain this

Read these before changing the shape of anything here.

- **Protocols are deliberately not `runtime_checkable`** (`4e695f12`,
  `efbf736c`): "isinstance against a Protocol only checks that method names
  exist, which is the stringly-typed dispatch this is removing."
- **An optional method fetched with `getattr` was removed from this exact
  method family** (`877cc945`): "`predict_outcome_proba` joins the TeamModel
  protocol instead of being fetched with getattr at its one call site." The two
  sanctioned patterns are a shared derivation helper the model calls
  (`outcome_proba_from_scores` in `team_models/scorelines.py`) and one named
  accessor with a documented fallback (`progress_total` in
  `optimization/protocols.py`). Prefer moving the decision into the table
  factory, which removes the runtime branch altogether.
- **The fit-data TypedDicts exist so mypy checks both ends** (`680862d3`): the
  assembler and the model reading it. Do not move feature assembly into each
  model. Extend the shared TypedDict with `NotRequired` keys, as
  `TeamFitData.team_covariates` already does.
- **Frozen request objects are the shape for these protocols** (`680862d3`):
  "The optimisation protocols have taken frozen request objects since they were
  written; this brings the prediction side level." Every new method below takes
  one request object, like `TransferRequest`. This also settles the argument-
  order convention for them.
- **There is deliberately no function that builds a whole pipeline from flags**
  (`4e695f12` removed `AIrsenalPipeline.from_names`; CLAUDE.md restates it). The
  CLI commands hand-build their pipeline block on purpose, so editing each of
  them is expected churn rather than a violation.
- **`--set-team`/`--set-player` string knobs were dropped on purpose**
  (`4e695f12`). Finer-grained configuration is done by constructing the
  component in Python and passing the object.
- **A named non-default component starts from its own settings** rather than
  being handed knobs it never asked for - which is why a team model that does no
  time weighting rejects `--epsilon` instead of ignoring it.

`alpha` in `PlayerFitData` is **not** a stray hyperparameter: it is an
empirical-Bayes prior derived from the fit window
(`features.get_empirical_bayes_estimates`) and used by two of the three player
models. It stays. Only its docstring changes, to describe the quantity rather
than mandate a Dirichlet.

## Phase 0 - the measurement gate

Every later phase needs a number that says whether it changed behaviour, and
today a non-probabilistic model cannot produce one.

- Add `score_points_predictions(...) -> PointsScore` (MAE, RMSE, rank
  correlation) to `prediction/evaluation.py`, reading `PlayerPrediction` rows
  for a tag against realised `PlayerScore.points`, plus `backtest_points(...)`
  walking the season forward like the two existing backtests.
- Record a baseline: `backtest_team_model` and `backtest_player_model` for every
  table entry over a fixed season and gameweek range, and one `airsenal replay`
  with a fixed `random_state`. These numbers are the regression oracle for
  phases 1-4.
- Guardrail: a test proving the new scorer works on a stub model that produces
  no probabilities at all.

Additive; no protocol changes. **Gate for phases 1-4: these numbers do not move**
(within float tolerance), and each commit message states the before and after.

### Done

`PointsScore`, `score_points_predictions` and `backtest_points` are in
`prediction/evaluation.py`; `get_predictions_for_gameweeks` in
`db/queries/predictions.py` reads the rows back per fixture. The guardrail is
`tests/prediction/test_points_evaluation.py`, which builds predictions by hand
and constructs no model at all - the property phases 1-4 rely on.

`backtest_points` differs from its two siblings in writing to the database: it
runs the real prediction stage per gameweek, under a tag prefixed
`Backtest_<season>_GW<gameweek>_`. Point it at a copy.

`tools/baseline_scores.py` regenerates everything below. Player models are
scored one position at a time, because `numpyro` cannot be fitted for
goalkeepers at all (see the phase 2 note).

### The baseline

Season 2526, gameweeks 5-30, horizon 1, against a copy of the live database,
recorded 2026-09-06:

```
uv run python tools/baseline_scores.py --season 2526 \
    --first-gameweek 5 --last-gameweek 30 --points
```

Held-out mean log probability, higher is better:

| model | score | observations |
|---|---|---|
| `team:extended` | -2.87858 | 261 |
| `team:neutral` | -2.88370 | 261 |
| `team:constant` | -4.79579 | 261 |
| `team:random` | -4.93000 | 261 |
| `player:conjugate` GK / DEF / MID / FWD | -0.03721 / -0.44705 / -0.75095 / -0.92684 | 402 / 2092 / 2803 / 717 |
| `player:numpyro` GK / DEF / MID / FWD | - / -0.44613 / -0.74992 / -0.92639 | - / 2092 / 2803 / 717 |
| `player:constant` GK / DEF / MID / FWD | -0.36981 / -0.50942 / -0.77129 / -1.08494 | 402 / 2092 / 2803 / 717 |

The whole points calculation with the default models, lower is better for the
errors and higher for the correlation:

| metric | value |
|---|---|
| mean absolute error | 0.9052 points |
| mean absolute error, players who appeared | 2.0921 points |
| root mean squared error | 1.8836 points |
| mean within-gameweek rank correlation | 0.8157 |
| observations | 20318, of which 7884 appearances |
| skipped | 0 |

**Watch `mean_absolute_error_played`, not just `mean_absolute_error`.** Six
observations in ten are a non-appearance, and 88% of those are predicted at
exactly zero - `points.py` returns 0.0 for a player with no recent minutes or an
injury flag - so they are correct for free and carry only 10.3% of the total
absolute error. They do not dominate the error; they dilute the mean, which is
why the headline is 0.905 against 2.092 for the players a squad is actually
picked from. A change that only affects starters moves the headline by about
0.39 of what it moves the played-only figure.

Nothing was skipped, so for a played gameweek the predictions and the
performances cover each other exactly. A non-zero `n_skipped` in a later phase
means something stopped being predicted.

## Scoring the parts, not just the total

Phase 0 scores one number per player per fixture. Four things are worth scoring
separately, and they need different amounts from a model:

| | what it asks | what it needs from the model |
|---|---|---|
| 1 | was the minutes prediction right? | an expected minutes, or a distribution over them |
| 2 | given the minutes actually played, was the goal-involvement share right? | an involvement **rate**, per 90 or per goal |
| 3 | what was the error in the points overall? | a number - phase 0, done |
| 4 | what was the error in each component? | a value per component, ideally re-askable at given minutes |

**(2) mostly exists already.** `score_player_model` conditions on actual
minutes and actual team goals - `player_outcome_probability` scales the fitted
rates by `min(minutes, 90) / 90` and evaluates the multinomial against what the
player actually did. What is missing is the error form of it, for a model whose
involvement is not a probability.

### The breakdown belongs on the prediction, not on the model

The tension: phase 4 makes `PointsModel` the seam, and a model behind it need
not have a minutes model or components inside it at all. Requiring four
sub-models to be exposed would undo what phase 4 is for.

So `PointsPrediction` carries an *optional* breakdown - what the model can
report about how it got there, not a structure it must have:

```python
@dataclass(frozen=True)
class PointsPrediction:
    expected_points: float
    expected_minutes: float | None = None
    involvement: Involvement | None = None  # rates, not points
    components: Mapping[str, float] | None = None
```

One scorer reads whatever is present and scores each part it finds. An
end-to-end regressor reports only `expected_points` and is scored on (3) alone;
a model that decomposes gets all four; a monolithic xG model that can still say
"I expect this player to play 60 minutes" gets (1) and (3) without having a
minutes model. Nothing is required, and everything reported is scored.

Two precisions this shape forces:

- **(2) needs a rate, not a number.** To condition on the minutes actually
  played, the scorer substitutes them itself - so the breakdown must carry
  `prob_score`/`prob_assist`, which are already per-goal rates, rather than the
  attacking points that came out of them.
- **(4) has two grades.** A *reported* component value is conditional on the
  model's own minutes prediction, so its error mixes in the minutes error. To
  decondition it, the component must be re-askable at given minutes - which is
  exactly `FittedComponent.expected_points(request)` from phase 3. So a
  reporting model gets the coarse version and a component model gets the sharp
  one.

### The observable side is exact

Checked on the live database: a realised FPL score decomposes into these
components and reconstructs `PlayerScore.points` **exactly**, for 27022 of
27022 performances in 2425 and 29747 of 29747 in 2526. So (4) has a ground
truth, and `actual_component_points()` is worth writing in phase 3 with the
reconciliation as its guardrail test - it would catch a divergence between
AIrsenal's scoring rules and FPL's the moment one appeared.

Two things that reconciliation turned up:

- **Three scoring events have no component at all**: own goals, penalties saved
  and penalties missed. Together they are about 0.5% of all points awarded (in
  2526: 40 own goals, 11 penalties saved, 15 missed). Small enough to leave
  unmodelled, but the component scorer needs a `residual` bucket rather than
  silently attributing them to a component that did not earn them. Note
  `game/scoring.py` has `points_for_own_goal` but no penalty-save or
  penalty-miss constant.
- **`MNG` is a position AIrsenal does not model.** FPL ran managers as a
  purchasable position in 24/25 only and then discontinued them, which is why
  322 performances in 2425 carry it and none in 2526. Correct data, not a bug -
  but `Position` has no `MNG` member, so anything indexing a per-position dict
  by the raw attribute string raises `KeyError` on those rows. A component
  scorer has to skip them.

### Metrics

For (1), minutes are not usefully continuous: what FPL pays for is the bands
`0` / `1-59` / `60+` - appearance points, and clean-sheet eligibility at
`MIN_MINUTES_FULL`. So score mean absolute error in minutes *and* accuracy over
the three bands; once phase 1 makes minutes a distribution, the log probability
of the actual band reuses `ModelScore` directly.

For (2) and (4), mean absolute error against the observable, split by
appearance as `PointsScore` already does.

### Where each one lands

Each phase adds the scorer its own structure makes possible, so a phase can be
judged by the thing it changed rather than only by the total:

- phase 1: `score_minutes_model` and `backtest_minutes_model` for (1)
- phase 2: the error form of (2), while the player protocol is open anyway
- phase 3: `actual_component_points`, per-component scoring for (4), and the
  reconciliation guardrail
- phase 4: the optional breakdown above, with the three scorers behind it

## Phase 1 - minutes becomes a model that owns its weights

- `MinutesModel` protocol returning a frozen `MinutesDistribution` (values plus
  weights summing to one). `RecentMinutesModel` reproduces today exactly:
  uniform weights over the same three values, with the
  `estimate_minutes_from_prev_season` fallback moving inside it.
  `get_recent_minutes_for_player` stays as the query it is.
- New `prediction/minutes_models/` package: `MINUTES_MODELS`,
  `build_minutes_model`, `--minutes-model`.
- `points.py` replaces its loop-and-divide with an explicit weighted
  expectation. This is the structural prerequisite for phase 3.
- Guardrails: `TABLES` in `tests/test_component_tables.py`; a new
  `tests/e2e/test_minutes_models.py` parametrized over the table;
  `docs/adding-a-model.md` ("five things are pluggable"); the file map in
  `docs/architecture.md`.

Also add `score_minutes_model` here - evaluation (1) above. A minutes model is
the first component whose prediction has its own observable, so it is the first
phase that can be judged by the thing it changed.

Independent of the protocol split, and it de-risks the `points.py` rewrite that
follows. Behaviour-preserving. ~1 commit.

### Done

`MinutesDistribution`, `MinutesRequest` and the `MinutesModel` protocol are in
`prediction/protocols.py`; `prediction/minutes_models/` holds the table and
`RecentMinutesModel`, reachable as `--minutes-model` from `predict`, `run` and
`replay`, and recorded in a replay's `config` block. `prediction/minutes.py` is
untouched: `get_recent_minutes_for_player` is a query, and the model calls it.

Three things came out differently from the sketch above:

- **`MinutesDistribution.expectation` carries the weighting**, so the points
  calculation reads `minutes.expectation(points_for_minutes)` and no caller
  averages anything. `probability_between` is there for scoring a band.
- **The empty-distribution check moved into the type.** `points.py` used to
  raise "Recent minutes is empty" on what the query returned; the invariant is
  now `__post_init__`'s, so every model gets it, along with weights summing to
  one and no negative values.
- **`fixtures_behind` and `min_fixtures_behind` left `points.py`** and became
  `RecentMinutesConfig`. No caller had ever passed either, so they were
  configuration masquerading as arguments - the lookback is now the model's own
  business, which is what made this phase a table entry rather than a rewrite.
  The fallback to last season's minutes stays inside
  `get_recent_minutes_for_player` where it already was; the sketch above said it
  would move, and moving it would have changed behaviour.

**The gate held.** Re-running the phase 0 points baseline over the same window:

| metric | phase 0 | phase 1 |
|---|---|---|
| mean absolute error | 0.905191 | 0.905191 |
| mean absolute error, appeared | 2.0921 | 2.092136 |
| root mean squared error | 1.883554 | 1.883554 |
| mean rank correlation | 0.815694 | 0.815694 |

Per prediction the two versions are not bitwise identical: about a quarter of
rows differ, by at most 9.5e-07. That is `sum(w * f(m))` rounding differently
from `sum(f(m)) / n`, not a change of behaviour - the team model's
probabilities arrive from jax in single precision, and 9.5e-07 is 2^-20. Worth
knowing for later phases: **the gate is the aggregate to six figures, not row
equality**, because any reassociation of the same arithmetic will move the last
bits.

### The minutes baseline, and what it says

`RecentMinutesModel` over season 2526, gameweeks 5-30 - the number any new
minutes model has to beat:

| metric | at phase 1 | with availability |
|---|---|---|
| mean absolute error | 13.3169 minutes | 9.2447 minutes |
| band accuracy | 0.7357 | 0.8239 |
| mean log probability | -3.4714 | -2.2553 |
| given no chance at all | 0.1187 | 0.0761 |

The second column is after availability moved into the model, at the end of
phase 4 - see below. Compare against that one.

The last row is the finding. A sample of recent appearances gives **zero**
probability to any band it did not sample, so a player who started three times
and is then benched was, according to the model, doing something impossible.
That happens to 11.9% of performances, and since `MIN_PROBABILITY` floors each
one at a log of about -27.6, those alone account for 3.3 of the -3.47. Band
accuracy of 0.74 and a log probability that bad are the same fact seen twice:
the point estimate is reasonable and the distribution is not a distribution.

So the metric to move first is `impossible_fraction`, not
`mean_absolute_error`. A model that put even a little weight on a benching
would improve the log score enormously while barely touching the error in
minutes - and it is the *distribution* the points calculation now consumes, via
`expectation`, so that weight would reach the predicted points.

## Phase 2 - separate "predicts a mean" from "predicts a distribution"

    TeamModel          : teams, fit, add_new_team, predict_expected_goals
    ScorelineTeamModel : + predict_score_n_proba, predict_outcome_proba

The table does the lifting, not a runtime check. `TEAM_MODELS` is annotated
`Callable[..., ScorelineTeamModel]`, and a mean-only model registers wrapped -
`PoissonScorelines(XGTeamModel())`, a new adapter beside
`outcome_proba_from_scores`. mypy checks the entry on the line you add it on;
`get_goal_probabilities_for_fixtures` and `score_team_model` simply require
`ScorelineTeamModel`. No `hasattr`, no branch, and the wrapped model's Poisson
pmf is a real predictive distribution, so it backtests honestly. This is
`4e695f12`'s own reason for writing table entries as small functions.

One finding from phase 0 sharpens this. `numpyro` cannot be fitted for
goalkeepers *at all*: no goalkeeper scores in a normal window, so `alpha`'s
first concentration is zero and the Dirichlet is improper, while `conjugate`
fits the same position from the same `alpha` quite happily. So the shared fit
data does carry a model-specific requirement after all - not in `alpha` existing,
but in a prior being handed to a model at all. A model that needs a proper
Dirichlet should build one from the counts in `y` itself. Worth doing here,
where the player protocol is already being changed.

The player side takes the same shape, and finishes the `680862d3` migration:
`get_probs` becomes a typed frozen return, with `ProbabilisticPlayerModel`
adding a posterior sampler (conjugate and numpyro both have one). Add
`NotRequired` xG/xA keys to `PlayerFitData` here too.

Guardrails: the method tuples in `tests/test_component_tables.py`; the two
parametrized e2e model tests; a new test that a mean-only model reaches the
points calculation through its adapter. ~2 commits, team then player.

Out of scope on purpose: `predict_scoreline_proba` for correlated scorelines.
Phase 2 makes it possible, but it is a modelling change that moves the numbers,
and mixing it into a refactor phase destroys the gate.

### Done

**Three protocols, not two.** The sketch above had `ScorelineTeamModel` extend a
`TeamModel` that already predicted a mean, which would have made all four
shipped models implement `predict_expected_goals` for a method with one caller
that never sees them. Instead:

| protocol | adds | who needs it |
|---|---|---|
| `TeamModel` | `teams`, `fit`, `add_new_team` | `add_new_teams_to_model` - fitting a model does not require knowing what it predicts |
| `ScorelineTeamModel` | `predict_score_n_proba`, `predict_outcome_proba` | the points calculation, `score_team_model`, `outcome_proba_from_scores` |
| `ExpectedGoalsTeamModel` | `predict_expected_goals` | `PoissonScorelines`, and nothing else |

No shipped model changed. `TEAM_MODELS` is annotated
`Callable[..., ScorelineTeamModel]`, and `PoissonScorelines` in
`team_models/scorelines.py` wraps a mean-only model at its table entry - so
`get_fitted_team_model` could stay plainly typed rather than generic: **the
table only holds scoreline models, because a mean-only one arrives already
wrapped.** mypy found every consumer that needed narrowing, which is the
argument for the split in one line.

Neither `constant` nor `random` could become mean-only models, incidentally -
both hold a pmf natively, and `constant`'s uniform scoreline is the deliberate
null baseline. So `PoissonScorelines`' first real user will be the xG model; for
now its callers are `tests/prediction/test_scorelines.py` and the worked example
in `tests/e2e/test_team_models.py`, which fits a mean-only model and runs it
through the whole points calculation.

**The player side** finishes the `680862d3` migration: `get_probs` is now
`predict_involvement`, returning a frozen `PlayerInvolvement` that checks the
three shares sum to one per player rather than claiming it in a docstring. The
name says share rather than probability, because that is all the points
calculation needs.

`score_involvement_error` is evaluation (2)'s error form - conditioned on the
minutes actually played and the goals the team actually scored, so it measures
the share alone. `team_goals_in` came out of `score_player_model` while it was
open, since both scorers needed it.

**Three things deliberately not done here:**

- **`ProbabilisticPlayerModel`** - a posterior sampler has no consumer until
  phase 5 widens the output boundary. Adding the protocol now would be a
  contract nothing checks and nothing calls.
- **`NotRequired` xG/xA keys on `PlayerFitData`** - nothing populates them yet.
  `TeamFitData.team_covariates` already shows the shape when a model wants
  them, and `process_player_data` can add them in the same commit as the model
  that reads them.
- **The Dirichlet-prior fix** (a model that needs a proper prior building it
  from the counts in `y` rather than receiving `alpha`). It would make `numpyro`
  fittable for goalkeepers, which is a *modelling* change: it moves that model's
  own scores. Doing it inside a refactor phase would break the gate that makes
  the refactor safe. It wants its own commit, with `backtest_player_model`
  before and after - the phase 0 baseline table has the numbers to beat.

**The gate held exactly.** Nothing here changes any arithmetic, so unlike phase
1 the predictions are bitwise identical: MAE 0.905191, appeared 2.092136, RMSE
1.883554, rank 0.815694 - all four unchanged over season 2526 GW5-30.

## Phase 3 - point components become a pluggable kind

```python
class FittedComponent(Protocol):
    def expected_points(self, request: ComponentRequest) -> float: ...
```

`ComponentRequest` is frozen and carries the player, the fixture, the position,
the team's score and concede distributions, the fitted involvement and the
minutes value - so a component asks for what it needs, and the seven positional
pandas parameters go. `appearance`, `attacking` and `defending` become
components alongside bonus, cards, saves and def_con; `point_components.py`
becomes a package with a table.

**Keep `PointsConfig` and the four CLI flags.**
`tests/prediction/test_points_config.py` exists because these flags were
"offered, accepted and then dropped" once already. `PointsConfig` gains a method
resolving to a component list; the flags and that parametrized test survive
unchanged.

Add `actual_component_points` and the per-component scorer here - evaluation (4)
above - with the reconciliation as its guardrail.

Behaviour-preserving: the component sum equals today's total within tolerance on
the seeded database. ~2 commits.

### Done

`point_components.py` is now a package with one module per part of a score and
`POINT_COMPONENTS` in its `__init__.py` - the seventh pluggable kind. Three
components are rules (`appearance`, `attacking`, `defending`) and four are
fitted averages (`bonus`, `cards`, `saves`, `def_con`), and all seven answer the
same two methods.

`calc_predicted_points_for_player` went from seven positional pandas parameters
to one `Sequence[PointComponent]`, and `points.py` from 273 lines to 137: it now
builds a `ComponentRequest` and sums whatever components the run was given.
`get_attacking_points` and `get_defending_points` moved into the components that
own them.

Two things worth noting about the shape:

- **`PointsConfig` kept its four flags**, as planned, and gained
  `component_names()`. The flags, `cli/predict.py` and
  `tests/prediction/test_points_config.py` are unchanged in meaning - the config
  moved to `point_components/__init__.py`, next to the table it now resolves
  against.
- **`make_predictedscore_table` takes `components=` too**, for a component no
  table knows about. Without it the kind would be pluggable only by name, which
  is not what the other six promise. There is deliberately no pipeline field:
  composing components belongs to phase 4's points model, and two ways to say it
  is what `4e695f12` deleted `from_names` for.

**The observable side is exact, in the shipped code.** `actual_component_points`
breaks a realised score into the same names, plus a `residual` bucket, and
reconstructs `PlayerScore.points` for **27022 of 27022** performances in 2425
and **29747 of 29747** in 2526. `game/scoring.py` gained
`points_for_penalty_save` and `points_for_penalty_miss` for it, which it had
been missing. Where the points actually go, in 2526:

| component | points | share |
|---|---|---|
| appearance | 19305 | 56% |
| attacking | 7653 | 22% |
| defending | 3336 | 10% |
| def_con | 2834 | 8% |
| bonus | 2415 | 7% |
| saves | 448 | 1% |
| cards | -1554 | -5% |
| residual | -55 | -0.2% |

Which reframes what is worth predicting well. Over half of all points are for
turning up, so the minutes model from phase 1 governs more of the total than
anything else - and `def_con`, in its first season, is already worth more than
bonus.

**The per-component scorer is not here.** A component's expected points are
conditional on the model's own minutes prediction, and deconditioning them means
re-asking each component at the minutes actually played - which needs the
prediction to carry its breakdown. That is phase 4's `PointsPrediction`, so the
scorer lands with it; the ground truth it needs is in place now.

**The gate held exactly**: MAE 0.905191, appeared 2.092136, RMSE 1.883554, rank
0.815694, all unchanged. The components sum in the same order the old function
added them in, so even the rounding is identical.

## Phase 4 - `PointsModel` as the top seam

By now `points.py` is a loop over components with injected team, player and
minutes models, so this is extraction and naming rather than new machinery.

- `prediction/points_models/`: `PointsModel` and `FittedPointsModel` (taking a
  frozen `PointsRequest`), `ComponentPointsModel` holding today's stack, a
  table, `--points-model`. An end-to-end xG model becomes a table entry.
- **The decision to make:** `8d2a9895` established
  `AIrsenalPipeline(team_model=..., player_model=...)`, and
  `docs/adding-a-model.md` documents it. Replacing those two fields with
  `points_model` re-opens that shape. Recommended anyway: the alternative -
  keeping both fields *and* adding `points_model`, erroring if both are given -
  is two ways to say one thing, which is what `4e695f12` deleted `from_names`
  for. The cost is one edit in each hand-built CLI pipeline block.
- `--team-model` and `--player-model` keep working by constructing
  `ComponentPointsModel(...)` in the CLI, one visible call per component.
  Combining them with a non-component `--points-model` raises `ConfigError` -
  the same rule as `--epsilon` on a model that does no time weighting.
- The pipeline still exposes four kinds, because team, player, minutes and
  components nest under the points model. ~2 commits.

### Done: the seam

`PointsModel` is `fit(PointsFitRequest)` and `predict(PointsRequest)`, returning
a `PointsPrediction`. `ComponentPointsModel` is the shipped one and
`prediction/points.py` is gone - its body is that model, and `make_prediction`
moved to `prediction/run.py`, which now loops players and fixtures and asks the
model.

**`AIrsenalPipeline` went from four prediction fields to one.** `team_model`,
`player_model`, `minutes_model` and `points` are replaced by `points_model`, as
recommended. The alternative - keeping them *and* adding `points_model` - would
have been two ways to say one thing, which is what `4e695f12` deleted
`from_names` for.

The flags all still work. `build_points_model(name, team_model=..., ...)` takes
the names and builds the component model from them, and naming a different
points model together with a part of the component one raises `ConfigError`
rather than ignoring it - the `--epsilon` rule.

**The team model is not a component.** It was worth asking: a component answers
in points and a team model answers in goals, and two components read it rather
than one, so it fills in the `ComponentRequest` instead of being an entry in the
list. A base-plus-subclass split would add a layer without removing one - the
base would be a points model that cannot predict attacking points, which is not
a useful thing to have.

`describe_pipeline` reads an optional `describe()` on the points model, the way
`progress_total` reads `num_increments`, so a replay's `config` block still
records which team, player and minutes models produced it. A points model
without one is named and left at that.

**The gate held exactly**: MAE 0.905191, appeared 2.092136, RMSE 1.883554, rank
0.815694.

### Done: the breakdown

`PointsPrediction` now carries optional `expected_minutes`, `involvement` (as
rates) and `components`, and `score_prediction_breakdown` scores whatever is
there. `backtest_breakdown` drives it over a season, scoring as it predicts
rather than reading rows back, so unlike `backtest_points` it writes nothing.

The four evaluations, from one call, on season 2526 GW5-30:

| | | |
|---|---|---|
| 1 | minutes | 9.24 minutes MAE |
| 2 | involvement | 0.174 goals, 0.173 assists, over 6014 matches the team scored in |
| 3 | points | 0.905 MAE, 2.092 over appearances, 1.884 RMSE, 0.816 rank correlation |
| 4 | components | attacking 0.411, defending 0.273, appearance 0.204, def_con 0.145, bonus 0.141, cards 0.084, saves 0.015 |

Three things those numbers say:

- **Attacking points carry the most error and appearance points the least**,
  even though appearance points are 56% of everything awarded and attacking
  only 22%. The biggest component is not the worst predicted one.
- **The minutes error was 9.24 here against 13.32 from `score_minutes_model`**,
  and that gap was the bug this phase found rather than a subtlety to be
  documented. See below.
- **The gate held through a second, independent path.** `backtest_breakdown`
  computes the totals in memory where `backtest_points` reads them back from the
  database, and the four figures agree exactly - MAE 0.905191, appeared
  2.092136, RMSE 1.883554, rank 0.815694, unchanged since phase 0.

A component's error is still conditional on the model's own minutes prediction,
which is the coarse grade of evaluation (4). The sharp grade - re-asking each
component at the minutes actually played - needs `FittedComponent` directly
rather than a reported number, so it belongs to whoever wants it rather than to
the generic scorer.

## Phase 5 - widen the database boundary (gated; probably defer)

`PlayerPrediction.predicted_points` is a scalar, so phases 1-4 only move *where*
a distribution collapses. Adding nullable variance or quantiles needs a
migration story and, more importantly, a consumer: `optimization/squad_score.py`
and the tree search read one number per player-gameweek. Do this only once there
is a model whose spread is worth optimising against - captaincy and bench order
are the obvious candidates. Until then, ending in a scalar is a real answer.

## Ordering

Phase 0 first, because it is the oracle. Phase 2 before phase 3, because the
component request carries the team model's output shape. Phase 1 is independent
and worth doing early: it is the assumption that prompted this.

Roughly eight commits plus the baseline, each shippable on its own, each with
the numbers in its commit message.

## Availability belongs to the minutes model

Found by the breakdown, the moment there were two numbers for one thing: the
points model reported 9.24 minutes MAE and `score_minutes_model` reported 13.32.
The difference was the injury and absence check, which sat in the points model
as a branch that zeroed a prediction the minutes model had already made.

Which is the wrong place for it twice over. A minutes model was being scored on
a prediction the run then overrode, so its number described something nobody
ran; and a *new* minutes model would inherit a filter it could not see, or lose
one it did not know to apply. An injured player plays no minutes - that is an
answer about minutes, and it belongs to the model that answers about minutes.

`is_absent` in `prediction/minutes.py` is the shared reading of it, next to the
other minutes queries, and `RecentMinutesModel` returns a point mass at zero for
a player it says is unavailable. `MinutesRequest` gained `fixture_gameweek`
alongside `root_gameweek` - a player can be back from injury later in a window,
so minutes are remembered per player per gameweek rather than per player. The
points model keeps only `expected_minutes == 0.0`, which now covers absence
without knowing what absence is.

**Points predictions did not move**: MAE 0.905191, appeared 2.092136, RMSE
1.883554, rank 0.815694. The filter moved; the arithmetic did not.

**The minutes model's own score improved a lot**, because it is now scored on
what the run actually uses: 13.32 to 9.24 minutes, band accuracy 0.7357 to
0.8239, and the log probability -3.47 to -2.26. A third of the outcomes the
model had called impossible were injured players it was confidently predicting
minutes for.

That the two numbers now agree exactly - the breakdown's minutes error and
`score_minutes_model`'s - is the point. Where they disagree again, something is
overriding a model's answer behind its back.

## The xG team model, which is what all of this was for

`XGTeamModel` in `team_models/xg.py` is the first model the loosened protocols
made possible: it predicts a mean and nothing else, and reaches the points
calculation through `PoissonScorelines`, wrapped in its own table entry. It is
an attack-and-defence rating fitted to the expected goals in past matches rather
than to the goals, by an alternating fit that rates each attack against the
defences it actually faced - which is what separates a good attack from an easy
schedule.

`TeamFitData` gained `home_expected_goals` and `away_expected_goals` as
`NotRequired` keys, populated by `get_result_dict` from a new
`get_expected_goals_by_fixture`. That is the extension point this plan pointed
at for the player side, used for real on the team side: no other model changed,
and one that assembles its own training data still type-checks.

**It beats the incumbent, on every season in the database.** Held-out mean log
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

**The whole gain is in defending, and attacking is slightly worse.** Which is
the sort of thing this plan was built to be able to see: xG is a better read on
how many a team concedes, and clean sheets and goals-conceded points follow
directly from that, while who takes a team's goals is the player model's job and
has not changed. A better team model on its own reaches predicted points mostly
through the defence.

It is not the default. `--team-model xg` selects it, and changing
`DEFAULT_TEAM_MODEL` would move every number in this document, so that is a
decision to take deliberately rather than as a side effect.

### Two obvious improvements, both nearly nothing

**Time weighting** was missing. It is there now, `exp(-epsilon * years ago)` as
the other models use, and `--epsilon` reaches it. Swept over 2425 and 2526 with
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

Built it, measured it, took it back out. The held-out log probability was
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
the residual scatter already 6-8% below the mean there is nothing for it to do -
which is why two names for one model was all it produced. The interesting
direction is the opposite one: a family that can be narrower than Poisson
(Conway-Maxwell-Poisson, or a binomial) might buy something. That is a modelling
change with no seam problem in the way, and it is not done.

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
season and filters on `current_team_only`. A promoted-team player has no such
history, so it returns `[0]`, the minutes are zero, and the points model
short-circuits. The same applies to every summer signing, which is part of why
only 44% of the rest are above zero either.

So the bias runs the opposite way to the worry: AIrsenal cannot over-pick
promoted-team players at gameweek 1 because it cannot pick them at all -
Sunderland's start to 25/26 was invisible to it. Fixing that is a decision about
what a player with no history should be assumed to play (their team's typical
minutes for the position, the FPL API's `chance_of_playing`, or their price as a
signal of whether they were bought to start), and it belongs to the minutes
model, which now owns availability. Until it is fixed, `promoted_like_bottom`
cannot matter at gameweek 1 whichever way it is set.

`tools/team_ratings.py` prints the fitted ratings, which is how the two newly
promoted teams were found sitting at a net 0.96 and 0.89 - almost exactly
average, as the worry supposed.
