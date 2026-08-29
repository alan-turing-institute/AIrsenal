# Adding a model or an algorithm

Five things are pluggable, and they compose into one object:

```python
AIrsenalPipeline(
    team_model=build_team_model("extended"),
    player_model=build_player_model("conjugate"),
    transfer_optimizer=TreeSearchOptimizer(),
    squad_optimizer=GeneticSquadOptimizer(),
    settings=PipelineSettings(...),
).run()
```

Each kind is a package. Its `__init__.py` holds a table mapping a name to a
factory, and a `build_*` function beside it turns a name plus the relevant CLI
flags into an object.

| kind | protocol | table and builder | CLI flag |
|---|---|---|---|
| player model | `PlayerModel` | `prediction/player_models/__init__.py`, `build_player_model` | `--player-model` |
| team model | `TeamModel` | `prediction/team_models/__init__.py`, `build_team_model` | `--team-model` |
| squad optimizer | `SquadOptimizer` | `optimization/squad_optimizers/__init__.py`, `build_squad_optimizer` | `--squad-optimizer` |
| transfer optimizer | `TransferOptimizer` | `optimization/transfer_optimizers/__init__.py`, `build_transfer_optimizer` | `--transfer-optimizer` |
| transfer strategy | `TransferStrategy` | `optimization/strategies/__init__.py` | none - the move picks it |

The protocols live in `prediction/protocols.py` and `optimization/protocols.py`,
and each names only the method that does the work.

## You do not have to register anything

`AIrsenalPipeline` takes *objects*, so a class defined in a notebook can be
dropped straight in:

```python
from airsenal.pipeline import AIrsenalPipeline, PipelineSettings

class MyTeamModel:
    ...  # fit, teams, add_new_team, predict_score_n_proba, predict_outcome_proba

AIrsenalPipeline(team_model=MyTeamModel(), settings=PipelineSettings(season="2425"))
```

The table is only how a *name on the command line* reaches an implementation.
`tests/e2e/test_pipeline_composition.py` pins this: a component no table knows
about works.

## Worked example: a new team model

### 1. Write the class

A team model has to answer five things - `teams`, `fit`, `add_new_team`,
`predict_score_n_proba` and `predict_outcome_proba`. It must also construct with
no arguments, defaulting its own config.

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

### 2. Add one line to the table

```python
TEAM_MODELS: dict[str, Callable[..., TeamModel]] = {
    "constant": _constant,
    "extended": _extended,
    "neutral": _neutral,
    "random": _random,
    "scoreline_average": _scoreline_average,   # <- this
}
```

The table is annotated with its protocol, so `mypy` checks your class fits at the
point you add it. If your model imports something expensive - jax, for instance -
make the entry a small function that imports inside itself, as the Dixon-Coles
entries do, so the cost is only paid when the model is actually built.

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
            TEAM_MODELS[name], season="2425", dbsession=session,
            gameweeks=range(5, 30), horizon=1,
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
