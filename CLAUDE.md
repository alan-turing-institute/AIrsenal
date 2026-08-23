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
uv run pytest tests/test_utils.py
# Single test:
uv run pytest tests/test_utils.py::test_function_name
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

**Check the package layering** (not run by pre-commit; run it after moving code
between packages):
```bash
uv run lint-imports
```

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

The packages form a one-way dependency chain, enforced by import-linter (see
`[tool.importlinter]` in `pyproject.toml`). Run `uv run lint-imports` after moving
code between them.

Bottom of the chain, depended on by everything:

- **`src/airsenal/core/`** — anything with no airsenal-specific dependencies: FPL's own
  rules (`scoring.py`, `enums.py`, `mappings.py`, `season.py`) and generic plumbing
  (`console.py`, `logging.py`, `caching.py`, `concurrency.py`, `dates.py`, `env.py`,
  `registry.py`, `data_files.py`). **If it does not import another airsenal module, it
  belongs here.**
- **`src/airsenal/data/`** — static historical FPL data (multiple seasons, used to seed
  the database); resolve paths with `airsenal.core.data_files.data_file()`, never with
  `__file__`
- **`src/airsenal/fetch/`** — everything that talks to an external source: the FPL API
  client and the Transfermarkt scraper
- **`src/airsenal/db/`** — `models.py` (all the SQLAlchemy tables), `queries/` (reading
  and writing them), `session.py`, `engine.py`, `admin.py`. This layer must not render
  output or make network calls.

The pipeline stages, each depending only on what is below it:

- **`src/airsenal/prediction/`** — team and player models, and the points they imply
- **`src/airsenal/squad/`** — the `Squad` and `CandidatePlayer` classes, and the state of
  the user's own entry
- **`src/airsenal/reporting/`** — rendering results: tables, plots, Discord posts
- **`src/airsenal/ingest/`** — filling the database from packaged data and the FPL API
- **`src/airsenal/export/`** — writing data back out (API dumps, DB dumps, attributes)
- **`src/airsenal/optimization/`** — the transfer search and the whole-squad builder
- **`src/airsenal/apply/`** — the only code that writes to the real FPL entry
- **`src/airsenal/pipeline/`** — top-level orchestration (`run`, `replay`)
- **`src/airsenal/cli/`** — Typer command definitions and CLI-only argument handling

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

### Swapping a model or an algorithm

Four things are pluggable, and they compose into one object:

```python
AIrsenalPipeline(
    team_model=build_team_model("extended"),  # prediction/protocols.py: TeamModel
    player_model=PLAYER_MODELS.create("conjugate"),  # PlayerModel
    transfer_optimizer=TRANSFER_OPTIMIZERS.create("tree_search"),  # TransferOptimizer
    squad_optimizer=SQUAD_OPTIMIZERS.create("genetic"),  # SquadOptimizer
    settings=PipelineSettings(...),
).run()
```

Each has a `Protocol` (in `prediction/protocols.py` or `optimization/protocols.py`)
and a `Registry` keyed by name. **Adding an implementation means writing a module
that registers itself; it should not require editing any call site.** If it does,
the seam is in the wrong place. `AIrsenalPipeline.from_names(...)` is the only place
a name becomes an object, and is what the CLI uses.

Settings belong to whichever component owns them - epsilon to the team model, the
GA config to the squad optimizer, thread count to the transfer optimizer - not to
the pipeline.

### Key modules

| File | Purpose |
|------|---------|
| `db/models.py` | SQLAlchemy ORM models (`Player`, `Fixture`, `PlayerScore`, `PlayerPrediction`, `Transaction`, etc.) |
| `db/session.py` | Lazily-created engine and the default session; nothing runs at import |
| `fetch/fpl_api.py` | FPL API client (uses `curl_cffi`); handles auth and data fetching |
| `prediction/team_models/dixon_coles.py` | BPL team-level match score predictions |
| `prediction/player_models.py` | Conjugate Bayesian and Numpyro player performance models |
| `prediction/points.py` | Turning fitted models into predicted points per fixture |
| `pipeline/run.py` | `AIrsenalPipeline`: the four swappable components plus the run settings |
| `optimization/run_transfers.py` | Fetching the squad, persisting suggestions and reporting around the search |
| `optimization/transfer_optimizers/` | One module per whole-window search, behind the `TransferOptimizer` protocol |
| `optimization/strategies/` | One module per way of choosing a gameweek's transfers, behind the `TransferStrategy` protocol |
| `optimization/squad_optimizers/` | One module per whole-squad builder, behind the `SquadOptimizer` protocol |
| `optimization/squad_ga.py` | The DEAP genetic algorithm the default squad optimizer wraps |
| `squad/squad.py` | `Squad` class: 15 players, formation/budget constraint checking |
| `core/enums.py` | `Position` and `Chip` |

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
