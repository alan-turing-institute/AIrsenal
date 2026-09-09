# Adding a model or an algorithm

Eight things are pluggable, and they compose into one object:

```python
AIrsenalPipeline(
    points_model=ComponentPointsModel(
        team_model=build_team_model("xg"),
        player_model=build_player_model("xg"),
        minutes_model=build_minutes_model("recent"),
    ),
    transfer_optimizer=TreeSearchOptimizer(),
    squad_optimizer=GeneticSquadOptimizer(),
    settings=PipelineSettings(...),
).run()
```

The pipeline itself takes three: a points model and the two optimizers. The team,
player and minutes models and the components are what `ComponentPointsModel` is
made of, not something every points model has - a model that regresses points
directly from its own features has none of them and is a table entry all the
same.

Each kind is a package. Its `__init__.py` holds a table mapping a name to a
factory, and a `build_*` function beside it turns a name plus the relevant CLI
flags into an object.

| kind | protocol | table and builder | CLI flag |
|---|---|---|---|
| points model | `PointsModel` | `prediction/points_models/__init__.py`, `build_points_model` | `--points-model` |
| player model | `PlayerModel` | `prediction/player_models/__init__.py`, `build_player_model` | `--player-model` |
| team model | `TeamModel` | `prediction/team_models/__init__.py`, `build_team_model` | `--team-model` |
| minutes model | `MinutesModel` | `prediction/minutes_models/__init__.py`, `build_minutes_model` | `--minutes-model` |
| point component | `PointComponent` | `prediction/point_components/__init__.py`, `build_point_component` | none - `PointsConfig` turns the optional ones off |
| squad optimizer | `SquadOptimizer` | `optimization/squad_optimizers/__init__.py`, `build_squad_optimizer` | `--squad-optimizer` |
| transfer optimizer | `TransferOptimizer` | `optimization/transfer_optimizers/__init__.py`, `build_transfer_optimizer` | `--transfer-optimizer` |
| transfer strategy | `TransferStrategy` | `optimization/strategies/__init__.py` | none - the move picks it |

The protocols live in `prediction/protocols.py` and `optimization/protocols.py`,
and each names only the method that does the work.

A point component is the one kind no flag selects by name. `PointsConfig` turns
the four fitted ones off - `--no-bonus`, `--no-cards`, `--no-saves`,
`--no-def-con` - and appearance, attacking and defending points are always
predicted, because without them there is no score to speak of. To predict with a
component of your own, pass it to `make_predictedscore_table(components=[...])`;
`tests/e2e/test_point_components.py` has a worked example.

`--team-model`, `--player-model`, `--minutes-model` and `--epsilon` describe
parts of the component model, so naming a different `--points-model` *and* one
of them is refused rather than half-honoured - the same rule as `--epsilon` on a
team model that does no time weighting.

## You do not have to register anything

`AIrsenalPipeline` takes *objects*, so a class defined in a notebook can be
dropped straight in:

```python
from airsenal.pipeline import AIrsenalPipeline, PipelineSettings


# fit, teams, add_new_team, predict_score_n_proba, predict_outcome_proba
class MyTeamModel: ...


AIrsenalPipeline(team_model=MyTeamModel(), settings=PipelineSettings(season="2425"))
```

The table is only how a *name on the command line* reaches an implementation.
`tests/e2e/test_pipeline_composition.py` pins this: a component no table knows
about works.

## Worked example: a new team model

### 1. Write the class

A team model has to answer five things - `teams`, `fit`, `add_new_team`,
`predict_score_n_proba` and `predict_outcome_proba`. It must also construct with
no arguments, defaulting its own config; the `--epsilon` flag reaches it through
the table entry in step 2, not through the constructor signature.

The last two are `ScorelineTeamModel`, and they are what the points calculation
needs: expected attacking points come from a multinomial over however many goals
the team scores, and a clean sheet is the probability of the opponent scoring
none. **If your model predicts only a mean** - an xG model fitted to a
continuous quantity has no natural distribution over goal *counts* - implement
`ExpectedGoalsTeamModel` instead, which is `teams`, `fit`, `add_new_team` and
`predict_expected_goals`, and wrap it in its table entry:

```python
TEAM_MODELS: dict[str, Callable[..., ScorelineTeamModel]] = {
    ...
    "xg": _xg,   # returns ConwayMaxwellScorelines(XGTeamModel())
}
```

`team_models/xg.py` is exactly that, and the worked example to copy: an attack
and defence rating fitted to expected goals, which has no distribution over goal
*counts* of its own.

There are two wrappers to choose from in `team_models/scorelines.py`, and they
differ only in how widely goals are taken to scatter around the mean:

- `PoissonScorelines` reads the mean as a Poisson, whose variance equals its
  mean. The tail above `MAX_GOALS` is piled onto the last count so the
  probabilities still sum to one.
- `ConwayMaxwellScorelines` makes that spread a parameter instead of an
  assumption, at `DEFAULT_GOAL_DISPERSION`; one is exactly Poisson and above one
  is narrower. It is what `xg` uses, because Premier League goals measure as
  narrower than Poisson. `tools/tune_goal_dispersion.py` re-derives the number.

Either way the table still promises a `ScorelineTeamModel`, so nothing
downstream has to ask which kind it was given, and mypy checks the wrapping on
the line you add it on.
`tests/e2e/test_team_models.py` has a worked example under
"a model that predicts only a mean".

A wrapper would otherwise be a dead end, so both expose two things. `.model` is
the model inside, which is how `tools/team_ratings.py` reads attack and defence
ratings off a wrapped model without knowing its class. `describe_component()`
names both, so a replay's `config` block records `ConwayMaxwellScorelines(
XGTeamModel)` rather than only the wrapper - two replays are worth comparing
only if you can see which model produced each, and the wrapper is the part they
would usually share.

`prediction/team_models/constant.py` is the smallest complete example. What
`fit` receives is `TeamFitData` in `prediction/protocols.py`: a `TypedDict`, so
your editor and `mypy` both know what is in it rather than you having to read the
function that assembles it.

```python
from airsenal.prediction.protocols import TeamFitData


class ScorelineAverageModel:
    """Every team scores the league average, whoever they are playing."""

    def __init__(self, config: MyConfig | None = None):
        self.config = config or MyConfig()
        self._teams: list[str] | None = None

    @property
    def teams(self) -> list[str] | None:
        return self._teams

    def fit(self, training_data: TeamFitData) -> "ScorelineAverageModel":
        ...
        return self
```

`outcome_proba_from_scores` in `prediction/team_models/scorelines.py` will
implement `predict_outcome_proba` for you if your model treats the two teams'
goal counts as independent.

A player model answers two things - `fit` and `predict_involvement`, which
returns a `PlayerInvolvement`: each fitted player's share of scoring, assisting
or neither for one of their team's goals. A share, not necessarily a
probability, so a model that reaches one without a posterior satisfies it too.
The three shares must sum to one per player, and `PlayerInvolvement` checks
that rather than trusting a docstring.

What `fit` receives is `PlayerFitData`, and it carries more than the goals: the
expected goals and assists of every (player, match), and the expected goals of
the player's whole team in it, which is what a share of one is a share of. They
are `NotRequired` keys, so a model that wants them must say so when they are
absent rather than quietly fitting to something else - `player_models/xg.py` is
the worked example, and is the conjugate model's Dirichlet update over that
count instead of the realised one.

### 2. Add a factory and one line to the table

`TEAM_MODELS` is the one table whose entries are not zero-argument: every factory
is called as `factory(epsilon=...)`, because `--epsilon` sets the time-weighting
decay rate. A model that does no time weighting should reject an epsilon it was
given rather than ignore it.

`--epsilon` reaches the *team* model only, and that is deliberate rather than an
omission. A player model has a time-weighting decay of its own - `epsilon` on
`ConjugatePlayerConfig` and `XGPlayerConfig` - but the two are tuned separately
and measure to different values, so one flag setting both would be wrong for
whichever it was not tuned for. `PLAYER_MODELS` is therefore name-only: a name
gets you the model with its own measured defaults, and varying a hyperparameter
means constructing the model in Python. `tools/tune_player_time_weighting.py`
keeps its own small table of `(epsilon, n_goals_prior) -> model` for exactly
that reason, and a new player model with hyperparameters worth sweeping should
be added to it as well as to `PLAYER_MODELS`.

```python
def _scoreline_average(*, epsilon: float | None = None) -> ScorelineTeamModel:
    if epsilon is not None:
        msg = "ScorelineAverageModel does not do time weighting"
        raise ValueError(msg)
    return ScorelineAverageModel()


TEAM_MODELS: dict[str, Callable[..., ScorelineTeamModel]] = {
    "constant": _constant,
    "extended": _extended,
    "neutral": _neutral,
    "random": _random,
    "xg": _xg,
    "scoreline_average": _scoreline_average,  # <- this
}
```

`ScorelineTeamModel` and not `TeamModel`: the table promises the kind that can
be predicted with, and `TeamModel` names only what every team model has in
common - fitting. A model that predicts a mean satisfies it through the wrapper
its own entry applies, which is why the annotation does not have to loosen for
`xg`.

The table is annotated with its protocol, so `mypy` checks your class fits at the
point you add it. Writing the entry as a small function rather than the class
itself is also how an expensive import is deferred - the Dixon-Coles entries
import jax inside themselves, so the cost is only paid when the model is actually
built.

### 3. There is no step three

Adding that line gets you:

- `--team-model scoreline_average` on `airsenal run`, `airsenal predict` and
  `airsenal replay`
- a build-and-protocol check, from `tests/test_component_tables.py`
- a real fit against the small seeded database, from
  `tests/e2e/test_team_models.py`, which parametrizes over `TEAM_MODELS`
- a scoring check, from `tests/e2e/test_evaluation.py`

If your model needs a setting no flag exposes, construct it in Python and pass
the object. `build_*` functions deliberately take only the flags that describe
their own kind, and a component named other than the default starts from its own
configuration rather than being handed knobs it never asked for.

## Is it any better?

Adding a model is half the job. `prediction/evaluation.py` is the other half.
Every scorer there is typed against the protocols, so a model that no table knows
about is scored exactly like one that ships.

```python
from airsenal.db.session import session_scope
from airsenal.prediction.evaluation import backtest_team_model
from airsenal.prediction.team_models import TEAM_MODELS

with session_scope() as session:
    for name in ("extended", "constant"):
        score = backtest_team_model(
            TEAM_MODELS[name],
            season="2425",
            dbsession=session,
            gameweeks=range(5, 30),
            horizon=1,
        )
        print(name, score.mean_log_probability)
```

`backtest_team_model` walks the season forward: for each gameweek it fits a fresh
model on the matches *before* it and scores the ones after. The number is a
held-out log probability - how much probability the model put on the results that
actually happened. Higher is better, and it is only comparable between models
scored over the same fixtures, which is why `ModelScore` carries the count.

`backtest_player_model` is the same for player models, and `score_team_model` /
`score_player_model` score an already-fitted model if you have one.

`backtest_breakdown` is the one to reach for once a points model exists. It
walks the season forward like the others, and scores every part of a prediction
the model was willing to report:

```python
score = backtest_breakdown(season="2526", dbsession=session, gameweeks=range(5, 31))
score.points.mean_absolute_error  # always
score.minutes.mean_absolute_error  # if it reported expected minutes
score.involvement  # if it reported goal shares
score.components["attacking"]  # per component, if it reported them
```

Everything but `points` is `None` when a model did not report it, and nothing is
held against a model for the parts it does not claim to have: an end-to-end
regressor is scored on its total alone. Fill in what you can on the
`PointsPrediction` you return and it gets scored - `involvement` in rates rather
than points, so the scorer can substitute the minutes actually played rather
than the ones you expected.

`actual_component_points` breaks a realised score into the same components a
run predicts, so each part has a ground truth to be scored against. It
reconstructs `PlayerScore.points` exactly for every performance in 2425 and
2526, which makes it a guardrail on the scoring rules as well: if AIrsenal's
ever drift from FPL's, `tests/prediction/test_actual_components.py` fails.

`score_involvement_error` is the error form of `score_player_model`, over the
same predictions: it asks only for a share, and conditions on the minutes
actually played and the goals the team actually scored, so it measures the share
alone with no minutes prediction mixed in.

Those two score one model by how much probability it put on what happened.
`backtest_points` scores the whole points calculation instead, by the error in
the points it predicted - which is the only number a model that is not
probabilistic can be judged by. `backtest_minutes_model` does the same for a
minutes model, in minutes and in the bands the scoring rules use. See
[prediction-seams-plan.md](prediction-seams-plan.md) for what each one can and
cannot tell you.

`tools/tune_team_time_weighting.py` and `tools/tune_player_time_weighting.py` are
grid sweeps built on exactly these functions, and are worth reading as longer
examples.

## Does it win?

A better log probability is not the same as a better squad. `airsenal replay`
plays a past season with your components and reports what the entry would have
scored:

```bash
uv run airsenal replay --season 2425 --team-model scoreline_average --output-dir runs/mine
uv run airsenal replay --season 2425 --team-model extended --output-dir runs/base
```

Each writes a JSON with `total_points`, `total_points_hit` and
`mean_absolute_error`, plus a `config` block recording which components produced
it. `replay_season` returns the same thing as a `ReplayResult` if you would
rather compare in Python than read the files back.

Replay takes the optimiser flags too - `--num-iterations`, `--num-generations`,
`--population-size`, `--num-free-transfers` - so a change to a search can be
measured the same way as a change to a model.
