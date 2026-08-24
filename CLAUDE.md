# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is AIrsenal

AIrsenal is a machine learning package for optimizing Fantasy Premier League (FPL) team selection and transfer decisions. It uses Bayesian statistical models to predict player/team performance, a greedy/brute-force approach to optimize transfers, and a DEAP genetic algorithm for initial whole-squad selection — all under FPL constraints (budget, squad size, position limits, chips, etc.).

## Commands

Always run Python with `uv run` or inside the virtual environment (`source .venv/bin/activate`).

**Install (including dev tools):**
```bash
uv sync --extra dev
```

**Run tests:**
```bash
uv run pytest tests
# Single test file:
uv run pytest tests/db/test_queries.py
# Single test:
uv run pytest tests/db/test_queries.py::test_function_name
```

**Lint and format:**
```bash
uv run ruff check --fix .
uv run ruff format .
```

**Type checking:**
```bash
uv run mypy
```

**Check the package layering:**
```bash
uv run lint-imports
```

`mypy` also runs as a pre-commit hook, over the files you are committing;
`lint-imports` does not, because it takes no filenames — a layering violation is an
edge between two modules, so it needs the whole import graph. It runs in CI, so run
it yourself after moving code between packages rather than finding out on the pull
request. CI runs both over everything.

**Pre-commit hooks:**
```bash
pre-commit install
pre-commit run --all-files
```

**Run the full pipeline (typical usage):**
```bash
uv run airsenal run
```

## Architecture

### Package layout

The packages form a one-way dependency chain, one package per layer, enforced by
import-linter (see `[tool.importlinter]` in `pyproject.toml`) and checked in CI, so
a wrong-direction import cannot merge — but **not** by pre-commit, so run `uv run
lint-imports` yourself after moving code between packages. The contract is
`exhaustive`, so a newly-created package has to be given a place in the chain
rather than silently escaping it. The order, top to bottom:

```
cli > pipeline > apply > optimization > export > ingest > reporting > squad >
prediction > db > remote > core
```

Bottom of the chain, depended on by everything:

- **`src/airsenal/core/`** — anything with no airsenal-specific dependencies: FPL's own
  rules (`scoring.py`, `enums.py`, `mappings.py`, `season.py`) and generic plumbing
  (`console.py`, `logging.py`, `caching.py`, `concurrency.py`, `copy.py`, `dates.py`,
  `env.py`, `lookup.py`, `data_files.py`). **If it does not import another airsenal
  module, it belongs here.**
- **`src/airsenal/data/`** — static historical FPL data (multiple seasons, used to seed
  the database); resolve paths with `airsenal.core.data_files.data_file()`, never with
  `__file__`
- **`src/airsenal/remote/`** — the one package that talks to the internet: the FPL API
  client, the Transfermarkt scraper, the Discord webhook poster, a resumable file
  downloader, and the error types they raise. **If it opens a socket, it belongs here** —
  enforced by a contract that forbids every other package from importing an HTTP client.
- **`src/airsenal/db/`** — `models.py` (all the SQLAlchemy tables), `queries/` (reading
  and writing them), `session.py`, `engine.py`. This layer must not render output or
  make network calls.

The pipeline stages, each depending only on what is below it:

- **`src/airsenal/prediction/`** — team and player models, and the points they imply
- **`src/airsenal/squad/`** — the `Squad` and `CandidatePlayer` classes, and the state of
  the user's own entry
- **`src/airsenal/reporting/`** — rendering results: tables, plots, Discord posts. It is
  below `optimization` on purpose, so it takes rows rather than a `Plan`.
- **`src/airsenal/ingest/`** — filling the database from packaged data and the FPL API
- **`src/airsenal/export/`** — writing data back out (API dumps, DB dumps, attributes)
- **`src/airsenal/optimization/`** — the transfer search and the whole-squad builder
- **`src/airsenal/apply/`** — the only code that writes to the real FPL entry
- **`src/airsenal/pipeline/`** — top-level orchestration (`run`, `replay`)
- **`src/airsenal/cli/`** — Typer command definitions and CLI-only argument handling.
  Options shared by more than one command live once, in `cli/options.py`.

Outside the package:

- **`tools/`** — dev one-offs, not packaged; install with the `tools` extra
- **`tests/`** — pytest tests, mirroring the package where there is enough to mirror

**Where does new code go?** If it has no airsenal imports, `core/`. Otherwise it belongs
to the pipeline stage that owns it — and a new subdirectory needs at least three modules
to be worth making.

### Data flow

1. **Database init** (`ingest/init_db.py`) — loads historical season data from
   `src/airsenal/data/` into a local SQLite database
2. **Database update** (`ingest/update.py`) — fetches current-season fixtures, results,
   and player attributes from the FPL API via `curl_cffi`
3. **Prediction** (`prediction/run.py`) — runs BPL (Bayesian Premier League) team models
   and player-level models to predict points; writes to `PlayerPrediction` table
4. **Optimization** (`optimization/run_transfers.py`) — uses a greedy/brute-force search
   to find optimal transfers; writes to `TransferSuggestion` table
5. **Apply** (`apply/transfers.py`, `apply/lineup.py`) — optionally posts transfers and
   lineup to the FPL API. NEVER run `apply/transfers.py` yourself whilst testing changes
   as this leads to irreversible changes to the actual AIrsenal FPL team entry.

`airsenal run` is the top-level orchestrator for steps 1-5.

### Adding or swapping a model or an algorithm

Five things are pluggable, and they compose into one object:

```python
AIrsenalPipeline(
    team_model=build_team_model("extended"),  # prediction/protocols.py: TeamModel
    player_model=build_player_model("conjugate"),  # PlayerModel
    transfer_optimizer=TreeSearchOptimizer(),  # optimization/protocols.py
    squad_optimizer=GeneticSquadOptimizer(),
    settings=PipelineSettings(...),
).run()
```

Each kind is a package, its `__init__.py` holds the table, and each has a
`Protocol` (in `prediction/protocols.py` or `optimization/protocols.py`) naming
the one method that does the work. The table maps a name to a zero-argument
factory:

| kind | protocol | table | CLI flag |
|------|----------|-------|----------|
| player model | `PlayerModel` | `prediction/player_models/__init__.py` | `--player-model` |
| team model | `TeamModel` | `prediction/team_models/__init__.py` | `--team-model` |
| squad optimizer | `SquadOptimizer` | `optimization/squad_optimizers/__init__.py` | `--squad-optimizer` |
| transfer optimizer | `TransferOptimizer` | `optimization/transfer_optimizers/__init__.py` | `--transfer-optimizer` |
| transfer strategy | `TransferStrategy` | `optimization/strategies/__init__.py` | none - the move picks it |

A transfer strategy is the one kind with no flag: which one runs is decided by
the move (`StrategySet.name_for`), not by the user.

**Adding an implementation is two steps:** write a class satisfying the protocol
that constructs with no arguments (default its config dataclass), then add one
line to the table. The tables are typed against their protocols, so mypy checks
the class fits at the point you add it, and `tests/test_component_tables.py`
picks the entry up automatically. Nothing else should need editing - if it does,
the seam is in the wrong place.

You do not have to register anything to use it: `AIrsenalPipeline` takes objects,
so a model defined in a notebook can be dropped straight in. The table is only how
a *name* on the command line reaches an implementation, and `lookup()` in
`core/lookup.py` is how a bad name becomes an error that lists the good ones.

Settings belong to whichever component owns them - epsilon to the team model, the
GA config to the squad optimizer, thread count to the transfer optimizer - not to
the pipeline. Only the settings that pre-date this are exposed as CLI flags
(`--epsilon`, `--num-generations`, `--population-size`, `--num-thread`,
`--num-iterations`), and each reaches only the component it describes: name a
different optimizer and it starts from its own defaults. Anything finer-grained
is set by constructing the component in Python.

Optionally, a component may also provide `num_increments()` to size its own
progress bar (see `progress_total` in `optimization/protocols.py`); without one
the bar runs indeterminate. A whole-squad optimizer is sized by
`SquadRequest.effort` - "search this hard, in whatever unit you count in" - which
is how one `--num-iterations` flag reaches both a standalone squad build and the
rebuild a wildcard or free hit does inside the transfer search.

### Key modules

| File | Purpose |
|------|---------|
| `db/models.py` | SQLAlchemy ORM models (`Player`, `Fixture`, `PlayerScore`, `PlayerPrediction`, `Transaction`, etc.) |
| `db/session.py` | Lazily-created engine and the default session; nothing runs at import |
| `remote/fpl_api.py` | FPL API client (uses `curl_cffi`); handles auth and data fetching |
| `remote/errors.py` | `RemoteError` and friends: what a failed call raises, so callers need not know the HTTP library |
| `prediction/team_models/dixon_coles.py` | BPL team-level match score predictions |
| `prediction/player_models/` | One module per player model, behind the `PlayerModel` protocol |
| `prediction/team_models/` | One module per team model, behind the `TeamModel` protocol |
| `prediction/points.py` | Turning fitted models into predicted points per fixture |
| `pipeline/run.py` | `AIrsenalPipeline`: the swappable components, the constraints and scoring, plus the run settings |
| `optimization/run_transfers.py` | Fetching the squad, persisting suggestions and reporting around the search |
| `optimization/plan.py` | `Plan` and `TransferSearchResult`: what a search produces |
| `optimization/squad_score.py` | What a squad is worth over a window, and `SquadScoringConfig` |
| `optimization/transfer_optimizers/` | One module per whole-window search, behind the `TransferOptimizer` protocol |
| `optimization/strategies/` | One module per way of choosing a gameweek's transfers, behind the `TransferStrategy` protocol |
| `optimization/squad_optimizers/` | One module per whole-squad builder, behind the `SquadOptimizer` protocol |
| `optimization/squad_optimizers/genetic_algorithm.py` | The DEAP genetic algorithm the default squad optimizer wraps |
| `squad/squad.py` | `Squad` class: 15 players, formation/budget constraint checking |
| `core/enums.py` | `Position` and `Chip` |
| `core/scoring.py` | FPL's own rules: points per event, `SQUAD_SIZE`, `MAX_FREE_TRANSFERS` |
| `cli/options.py` | The `Annotated` option aliases shared across commands |
| `core/lookup.py` | `lookup()` and `ConfigError`: turning a name into an implementation |

### Database

SQLite, default location: `$AIRSENAL_HOME/data.db` (configurable via `AIRSENAL_DB_FILE`
env var). SQLAlchemy v2.0+ ORM. The `dbsession` argument (defaulting to
`airsenal.db.session.get_session()`) is threaded through most functions.

### Configuration

Required env var: `FPL_TEAM_ID`. Optional: `FPL_LOGIN`, `FPL_PASSWORD`, `FPL_LEAGUE_ID`,
`AIRSENAL_DB_FILE`. Use `airsenal env set` to persist these under `AIRSENAL_HOME`.

### Prediction is single-threaded by design

`prediction/run.py` used to parallelize player predictions with a thread/process pool;
this was removed because jax deadlocks under multi-threading, and prediction is fast
enough without it. Don't reintroduce multi-threading/multiprocessing there (or in code
that calls jax-based models) unless the deadlock issue is independently resolved.

## Code conventions

- **Branch naming:** `feature/<issue>-<description>` or `bugfix/<issue>-<description>`; all new branches should be made from `develop`, and all pull requests should be made to merge into `develop`
- **Function argument order** (where applicable): other args → `player`/`player_id` → `position` → `team` → `tag` → `gameweek` → `season` → `fpl_team_id` → `dbsession` → `fetcher` → `verbose`
- **Season strings:** `"2122"` for the 2021/22 season
- **Positions and chips:** use the `Position` and `Chip` enums from `core/enums.py`, not bare strings (`"all"` is still a plain string where a position filter accepts it)
- Docstrings should follow numpydoc convention; type hints are encouraged
