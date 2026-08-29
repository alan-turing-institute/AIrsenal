# How AIrsenal is put together

Where things live, and why they live there. For what the code *does* at runtime -
the database schema and how points predictions are built - see
[how-it-works.md](how-it-works.md). For how to add a model or an algorithm, see
[adding-a-model.md](adding-a-model.md).

## The dependency chain

The package is thirteen subdirectories of `src/airsenal`, forming a one-way
chain. Each may import from the rows below it and never from the rows above.

```
cli            command definitions and argument parsing, nothing else
pipeline       orchestration: `run` and `replay`
apply          the only code that writes to the real FPL entry
optimization   the transfer search and the whole-squad builder
export         writing data back out
ingest         filling the database from packaged data and the FPL API
reporting      rendering results: tables, plots, Discord posts
squad          the Squad class and the state of the user's own entry
prediction     the models, the points they imply, and how to score a model
db             the tables, the queries and the session
remote         everything that talks to the internet, and nothing else
core           generic plumbing: logging, caching, the console, date parsing
game           the facts about FPL - it imports nothing at all
```

`game/` is last, and that is the point: what a position is, what a goal is worth,
what `"2122"` means. It imports nothing - not another airsenal package, not a
third-party library, not even a logger.

Four `import-linter` contracts in `pyproject.toml` enforce this, and
`uv run lint-imports` checks them:

| contract | what it stops |
|---|---|
| Stage packages form a one-way dependency chain | an import going back up the list. `exhaustive`, so a new package must be given a place rather than escaping the chain |
| The database layer does not reach for the FPL API | a query quietly becoming a live API call |
| The database layer neither renders nor talks to the network | `db` reaching for Rich, matplotlib or an HTTP client |
| Only the remote package talks to the network | any other package importing `curl_cffi`, `requests` or `bs4` |

Two things the contracts cannot see are checked by tests instead:
`tests/game/test_game_is_plain_python.py` walks each `game/` module's syntax tree
and allows only stdlib and sibling imports, and `tests/test_import_side_effects.py`
imports every module with sockets and SQLite blocked.

A wrong-direction import does not fail at run time, so run `uv run lint-imports`
yourself after moving code between packages. The pre-commit hook also runs it.

## Where new code goes

Three questions, in order:

1. **Is it a fact about Fantasy Premier League?** What a goal is worth, what a
   position is, how season strings work, what other data sources call a club -
   that goes in `game/`. Needing a logger or a dataframe is the signal that it is
   not a fact about the game.
2. **Is it generic Python machinery with no airsenal imports?** That goes in
   `core/`.
3. **Otherwise it belongs to the stage that owns it** - the list above. If it
   seems to belong to two stages, it goes in the lower one, or it is two
   functions.

Prefer a module in an existing package to a new subdirectory holding one file: a
directory tells the reader a category exists without telling them what is in it.

`CodingConventions.md` at the repository root is the canonical version of this
rule, along with the naming and argument-order conventions.

## The map

| file | what is in it |
|---|---|
| `game/enums.py` | `Position` and `Chip` |
| `game/scoring.py` | FPL's own rules: points per event, `SQUAD_SIZE`, `MAX_FREE_TRANSFERS` |
| `game/season.py` | which season it is, and how a season is written |
| `game/mappings.py` | what other data sources call clubs and positions |
| `core/lookup.py` | `lookup()` and `ConfigError`: turning a name into an implementation |
| `core/concurrency.py` | the fork the transfer search needs, and the handlers that make it safe |
| `core/console.py` | Rich console, tables, progress bars, and `confirm()` |
| `remote/fpl_api.py` | the FPL API client |
| `remote/errors.py` | `RemoteError` and friends, so callers need not know the HTTP library |
| `db/models.py` | every table in the database |
| `db/queries/` | reading and writing them, one module per subject |
| `db/session.py` | the lazily-created engine and the default session |
| `prediction/protocols.py` | `PlayerModel`, `TeamModel`, and the typed data each is fitted to |
| `prediction/features.py` | assembling the historical data the models are fitted to |
| `prediction/player_models/` | one module per player model, plus shared fitting and scaling |
| `prediction/team_models/` | one module per team model, plus shared fitting and scorelines |
| `prediction/point_components.py` | the fitted models for bonus, saves, cards and defensive contributions |
| `prediction/points.py` | turning fitted models into predicted points, and `PointsConfig` |
| `prediction/evaluation.py` | scoring a fitted model against what actually happened |
| `prediction/run.py` | filling the prediction table, and the tag that groups a run's rows |
| `squad/squad.py` | `Squad`: fifteen players and the rules they obey |
| `squad/state.py` | the state of the user's own entry, from the database and the API |
| `optimization/protocols.py` | `SquadOptimizer`, `TransferStrategy`, `TransferOptimizer` and their requests |
| `optimization/plan.py` | `Plan` and `TransferSearchResult`: what a search produces |
| `optimization/squad_score.py` | what a squad is worth over a window, and `SquadScoringConfig` |
| `optimization/transfer_optimizers/` | one module per whole-window search |
| `optimization/strategies/` | one module per way of choosing a gameweek's transfers |
| `optimization/squad_optimizers/` | one module per whole-squad builder |
| `pipeline/run.py` | `AIrsenalPipeline`: the swappable components and the run settings |
| `pipeline/replay.py` | replaying a past season, and the `ReplayResult` it scores |
| `cli/options.py` | the option aliases shared across commands |

Outside the package: `tests/` mirrors it where there is enough to mirror,
`tools/` holds dev one-offs (installed with the `tools` extra, and type-checked),
and `notebooks/` holds exploratory Jupyter notebooks.

## Two things that constrain future work

**Prediction is single-threaded on purpose.** Do not add threading or
multiprocessing to `prediction/run.py`, or to any code that calls a jax-based
model: jax deadlocks under multi-threading. Prediction is fast enough without it.

**The transfer search must fork.** `core/concurrency.py` forces the `fork` start
method on posix. The search hands its workers local progress callbacks, which
pickle cannot serialise, so under `spawn` - macOS's default - it does not run
slower, it fails. It can also only fork before jax has been initialised.
