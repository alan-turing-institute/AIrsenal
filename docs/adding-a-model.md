# Adding a model or an algorithm

Eight kinds of component can be swapped out, and they fit together like this:

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

The pipeline itself takes three components: a points model and the two optimizers.
The team, player and minutes models and the point components are the parts of
`ComponentPointsModel` specifically. Other points models don't need them: a model that
predicts points directly from its own features would have none of them, and is added
to its table in the same way.

Each kind of component has its own package. The package's `__init__.py` holds a table
mapping names to factory functions, and a `build_*` function that takes a name and the
relevant CLI flags and returns an object.

| kind | protocol | table and builder | CLI flag |
|---|---|---|---|
| points model | `PointsModel` | `prediction/points_models/__init__.py`, `build_points_model` | `--points-model` |
| player model | `PlayerModel` | `prediction/player_models/__init__.py`, `build_player_model` | `--player-model` |
| team model | `TeamModel` | `prediction/team_models/__init__.py`, `build_team_model` | `--team-model` |
| minutes model | `MinutesModel` | `prediction/minutes_models/__init__.py`, `build_minutes_model` | `--minutes-model` |
| point component | `PointComponent` | `prediction/point_components/__init__.py`, `build_point_component` | none - `PointsConfig` turns the optional ones off |
| squad optimizer | `SquadOptimizer` | `optimization/squad_optimizers/__init__.py`, `build_squad_optimizer` | `--squad-optimizer` |
| transfer optimizer | `TransferOptimizer` | `optimization/transfer_optimizers/__init__.py`, `build_transfer_optimizer` | `--transfer-optimizer` |
| transfer strategy | `TransferStrategy` | `optimization/strategies/__init__.py` | none - chosen by the type of move being searched |

The protocols are defined in `prediction/protocols.py` and `optimization/protocols.py`,
and each one only requires the methods that do the work.

Point components are the only kind that can't be chosen by name on the command line.
The four fitted components can be switched off with `--no-bonus`, `--no-cards`,
`--no-saves` and `--no-def-con`, which set `PointsConfig`. Appearance, attacking and
defending points are always predicted, since without them there isn't much of a score
left. To predict with a component of your own, pass it to
`make_predictedscore_table(components=[...])`; `tests/e2e/test_point_components.py`
has a worked example.

`--team-model`, `--player-model`, `--minutes-model` and `--epsilon` only apply to
`ComponentPointsModel`. If you choose a different `--points-model` and also pass one
of these flags, AIrsenal raises an error rather than silently ignoring the flag. For
the same reason, passing `--epsilon` to a team model that doesn't use time weighting
is an error.

## You don't have to register anything

`AIrsenalPipeline` takes *objects*, so you can pass it a class defined in a notebook
directly:

```python
from airsenal.pipeline import AIrsenalPipeline, PipelineSettings
from airsenal.prediction.points_models import ComponentPointsModel


# fit, teams, add_new_team, predict_score_n_proba, predict_outcome_proba
class MyTeamModel: ...


AIrsenalPipeline(
    points_model=ComponentPointsModel(team_model=MyTeamModel()),
    settings=PipelineSettings(season="2425"),
)
```

The tables only exist so that a *name on the command line* can be turned into an
object. `tests/e2e/test_pipeline_composition.py` checks that a component that isn't in
any table still works.

## Rules the protocols follow

Read these before changing a protocol:

- **Protocols are not `runtime_checkable`.** `isinstance` against a protocol only
  checks that the method names exist. mypy checks that implementations actually
  conform, wherever a table entry or a pipeline field is annotated with the protocol.
- **Don't use `getattr` at a call site to look for an optional method.** Where models
  differ in what they can do, handle it in one of three ways: in the table's factory
  function (the `xg` team model entry wraps a model that only predicts a mean), with a
  shared helper the models can call (`outcome_proba_from_scores`), or in a single named
  function with a documented fallback (`progress_total`, `describe_pipeline`).
- **Fitting data is a shared `TypedDict`** (`PlayerFitData`, `TeamFitData`), so mypy
  checks both the code that builds it and the models that read it. The data is
  assembled in one place rather than in each model. A model that needs extra data adds
  a `NotRequired` key, and raises a clear error if it is missing.
- **Each protocol method takes a single frozen request object**, such as
  `TransferRequest`, `MinutesRequest` or `PointsRequest`. This also avoids arguments
  being passed in the wrong order.
- **There is no function that builds a whole pipeline from CLI flags.** Each CLI
  command builds its own `AIrsenalPipeline`, so adding a field means editing each of
  them.
- **Settings that don't have a CLI flag are set in Python**: construct the component
  yourself and pass in the object, rather than adding more options to the CLI.
- **A non-default component uses its own default settings**, and isn't passed flags
  meant for another component. That's why a team model that doesn't use time weighting
  rejects `--epsilon` rather than ignoring it.
- **Whether a player plays is decided by the minutes model.** A player who is
  unavailable (`is_absent` in `prediction/minutes.py`) should be predicted zero minutes
  with probability one, and the points model only checks for
  `expected_minutes == 0`. A minutes model that ignores `is_absent` will predict
  minutes for injured players. The points model must not filter out unavailable
  players itself, or the minutes model would be evaluated on predictions that were
  never used.

## Worked example: a new team model

### 1. Write the class

A team model must implement five things: `teams`, `fit`, `add_new_team`,
`predict_score_n_proba` and `predict_outcome_proba`. It must also be possible to
construct it with no arguments, using default settings. The `--epsilon` flag reaches
it through the table entry in step 2, not through the constructor.

`predict_score_n_proba` and `predict_outcome_proba` make up the `ScorelineTeamModel`
protocol, and they are what the points calculation needs. Expected attacking points
come from a multinomial distribution over however many goals the team scores, and the
chance of a clean sheet is the probability of the opponent scoring none.

**If your model only predicts a mean number of goals** (for example, a model fitted to
expected goals, which are continuous and so don't give a distribution over whole
numbers of goals), implement the `ExpectedGoalsTeamModel` protocol instead. That needs
`teams`, `fit`, `add_new_team` and `predict_expected_goals`. Then wrap it in its table
entry:

```python
TEAM_MODELS: dict[str, Callable[..., ScorelineTeamModel]] = {
    ...
    "xg": _xg,   # returns ConwayMaxwellScorelines(XGTeamModel())
}
```

`team_models/xg.py` does exactly this, and is a good example to copy.

`team_models/scorelines.py` has two wrappers to choose from. They only differ in how
widely the number of goals varies around the mean:

- `PoissonScorelines` treats the mean as the rate of a Poisson distribution, whose
  variance equals its mean. Probability above `MAX_GOALS` goals is added to the
  `MAX_GOALS` count, so the probabilities still sum to one.
- `ConwayMaxwellScorelines` adds a dispersion parameter, set to
  `DEFAULT_GOAL_DISPERSION`. A dispersion of one is the same as a Poisson, and above
  one the distribution is narrower. The `xg` team model uses it, because Premier League
  goal counts vary less than a Poisson predicts. `tools/tune_goal_dispersion.py`
  recalculates the value.

Either way, the table entry is still a `ScorelineTeamModel`, so the code that uses it
doesn't need to know which kind of model it has, and mypy checks the wrapping on the
line where you add it. `tests/e2e/test_team_models.py` has an example under "a model
that predicts only a mean".

Both wrappers give access to the model inside them:

- `.model` is the wrapped model. `tools/team_ratings.py` uses it to read attack and
  defence ratings without knowing the wrapper's class.
- `describe_component()` names both the wrapper and the model, so a replay's `config`
  block records `ConwayMaxwellScorelines(XGTeamModel)` rather than just the wrapper.
  Without that, you couldn't tell which model produced two replays that used the same
  wrapper.

`prediction/team_models/constant.py` is the smallest complete example. `fit` is given
a `TeamFitData` (defined in `prediction/protocols.py`). It's a `TypedDict`, so your
editor and mypy know what's in it without you having to read the code that builds it.

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

If your model treats the two teams' goals as independent, `outcome_proba_from_scores`
in `prediction/team_models/scorelines.py` can implement `predict_outcome_proba` for
you.

### 2. Add a factory function and a line to the table

`TEAM_MODELS` is the only table whose factories take an argument: each one is called
as `factory(epsilon=...)`, where `--epsilon` sets the time-weighting decay rate. A model
that doesn't use time weighting should raise an error if it is given an epsilon, rather
than ignore it.

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

The table is annotated with `ScorelineTeamModel` rather than `TeamModel`, because
entries must be able to make predictions, and `TeamModel` only covers what every team
model has in common (fitting). Models that only predict a mean still fit the annotation
because their table entry wraps them.

Because the table is annotated with a protocol, mypy checks that your class conforms
when you add it. Writing the entry as a small function rather than using the class
directly also lets you delay expensive imports: the Dixon-Coles entries import jax
inside the function, so the cost is only paid when that model is used.

### 3. What you get

Adding that line gives you:

- `--team-model scoreline_average` on `airsenal run`, `airsenal predict` and
  `airsenal replay`
- a check that it builds and implements the protocol, from
  `tests/test_component_tables.py`
- a real fit against the small test database, from `tests/e2e/test_team_models.py`,
  which is parametrized over `TEAM_MODELS`
- a scoring check, from `tests/e2e/test_evaluation.py`

If your model needs a setting that no flag exposes, construct it in Python and pass in
the object. `build_*` functions only take the flags that apply to their own kind of
component.

### Differences for a player model

A player model implements `fit` and `predict_involvement`. `predict_involvement`
returns a `PlayerInvolvement`: for each fitted player, their share of scoring,
assisting, or neither, for each of their team's goals. These are shares rather than
strictly probabilities, so a model doesn't need a posterior distribution to produce
them. The three shares must sum to one for each player, which `PlayerInvolvement`
checks.

`fit` is given a `PlayerFitData`, which includes more than goals: the expected goals
and assists for every (player, match), and the expected goals of the player's whole
team in that match, which is the total a player's share is taken from. These are
`NotRequired` keys, so a model that uses them must raise an error when they are
missing rather than silently fitting to something else. `player_models/xg.py` is an
example: it is the conjugate model's Dirichlet update, applied to expected rather than
actual goals and assists.

`--epsilon` only applies to *team* models, on purpose. Player models have their own
time-weighting parameter (`epsilon` on `ConjugatePlayerConfig` and `XGPlayerConfig`),
but the best values for team and player models are different, so a single flag
setting both would be wrong for one of them. `PLAYER_MODELS` entries therefore take no
arguments: a name gives you the model with its own tuned defaults, and changing a
hyperparameter means constructing the model in Python.
`tools/tune_player_time_weighting.py` has its own small table of
`(epsilon, n_goals_prior) -> model` for this reason. If you add a player model with
hyperparameters worth tuning, add it there as well as to `PLAYER_MODELS`.

## Is it any better?

Adding a model is half the job. The other half is `prediction/evaluation.py`. Its
scoring functions are typed against the protocols, so a model that isn't in any table
is scored in the same way as the built-in ones.

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

`backtest_team_model` steps through the season: for each gameweek it fits a new model
on the matches *before* it and scores its predictions for the matches after. The
result is a held-out log probability, a measure of how much probability the model gave
to the results that actually happened. Higher is better. Scores can only be compared
between models evaluated on the same fixtures, which is why `ModelScore` includes the
number of fixtures.

The other functions:

- **`backtest_player_model`** does the same for player models. `score_team_model` and
  `score_player_model` score a model you have already fitted.
- **`backtest_breakdown`** is the one to use once you have a points model. It steps
  through the season in the same way, and scores every part of a prediction that the
  model reports:

  ```python
  score = backtest_breakdown(season="2526", dbsession=session, gameweeks=range(5, 31))
  score.points.mean_absolute_error  # always
  score.minutes.mean_absolute_error  # if it reported expected minutes
  score.involvement  # if it reported goal shares
  score.components["attacking"]  # per component, if it reported them
  ```

  Everything except `points` is `None` if the model didn't report it, and a model
  isn't penalised for parts it doesn't report: a model that predicts total points
  directly is only scored on its total. Anything you fill in on the `PointsPrediction`
  you return is scored. `involvement` is reported as rates rather than points, so the
  scorer can use the minutes the player actually played rather than the predicted
  minutes.
- **`actual_component_points`** splits an actual score into the same components a run
  predicts, so each component can be compared with what happened. It exactly
  reproduces `PlayerScore.points` for every performance in 2425 and 2526, so it also
  checks AIrsenal's scoring rules: if they drift from FPL's,
  `tests/prediction/test_actual_components.py` fails.
- **`score_involvement_error`** measures the error in a player model's shares, using
  the same predictions as `score_player_model`. It uses the minutes actually played and
  the goals the team actually scored, so the result only reflects the shares, not the
  minutes prediction.
- **`backtest_points`** scores the whole points calculation by the error in its
  predicted points, rather than by log probability. It's the only way to evaluate a
  model that doesn't give probabilities. Unlike the others it writes to the database,
  with tags starting `Backtest_<season>_GW<gameweek>_`, so run it on a copy.
- **`backtest_minutes_model`** scores a minutes model, both in minutes and in the
  minutes bands the scoring rules use (0, 1-59, 60+).

What each score can and can't tell you:

- **Use log probability to compare team models or player models.** It only compares
  models evaluated on the same observations.
- **Points error can't tell you whether a player model is better.** A player model
  only affects the attacking component, so it barely changes total points. Also, most
  performances score no attacking points, so a model that under-predicts goals and
  assists gets a *lower* points error. Use the log probability of the shares instead;
  [xg-models.md](xg-models.md) has the measurements.
- **Read `mean_absolute_error_played` alongside `mean_absolute_error`.** Most
  observations are players who didn't play and were correctly predicted zero points.
  They lower the average error without telling you anything about the players you'd
  actually pick.
- **A component's error depends on the model's own minutes predictions**, so it
  includes the minutes error. `involvement` is scored using the actual minutes played,
  which is why it's reported as rates.
- **A minutes model built from a sample of past appearances gives zero probability to
  a minutes band it hasn't seen**, and each such performance costs about 27 in log
  score. Read `impossible_fraction` alongside `mean_log_probability`.
- **A non-zero `n_skipped`** means predictions and actual performances didn't match up:
  something wasn't predicted, or wasn't scored.

Only `expected_points` is saved to the database. `PlayerPrediction.predicted_points` is
a single number per player per fixture, and it's all the optimizers
(`optimization/squad_score.py` and the tree search) use. So any uncertainty a model
predicts is currently unused. Captaincy and bench order are the obvious places it
could help.

`tools/tune_team_time_weighting.py` and `tools/tune_player_time_weighting.py` are grid
searches built on these functions, and are useful longer examples.

## Does it pick better teams?

A better log probability doesn't guarantee a better squad. `airsenal replay` plays
through a past season with your components and reports how many points the team would
have scored:

```bash
uv run airsenal replay --season 2425 --team-model scoreline_average --output-dir runs/mine
uv run airsenal replay --season 2425 --team-model extended --output-dir runs/base
```

Each run writes a JSON file with `total_points`, `total_points_hit` and
`mean_absolute_error`, plus a `config` block recording which components were used.
`replay_season` returns the same results as a `ReplayResult`, if you'd rather compare
them in Python.

Replay also accepts the optimizer flags (`--transfer-optimizer`, `--max-expansions`,
`--num-iterations`, `--num-generations`, `--population-size`, `--num-free-transfers`),
so changes to the search can be measured in the same way as changes to a model.

The default transfer optimizer, `auto`, searches the whole plan tree when it is small
and uses MCTS once the tree has more than twice as many nodes as the MCTS budget
(`--max-expansions`), which is about from a six-gameweek window on. To compare the
two searches on the same window, name each with `--transfer-optimizer`. A replay's
first squad is built from scratch by the genetic algorithm, so two replays start
from different squads. To compare searches, resume both from the same squad with
`--resume --gameweek-start 2`, each on a copy of a database that holds only that
replay's first-gameweek transactions.
