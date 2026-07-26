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
uv run pytest airsenal/tests
# Single test file:
uv run pytest airsenal/tests/test_utils.py
# Single test:
uv run pytest airsenal/tests/test_utils.py::test_function_name
```

**Lint and format:**
```bash
uv run ruff check --fix .
uv run ruff format .
```

**Type checking:**
```bash
uv run mypy airsenal/framework airsenal/scripts
```

**Pre-commit hooks:**
```bash
pre-commit install
pre-commit run --all-files
```

**Run the full pipeline (typical usage):**
```bash
uv run airsenal_run_pipeline
```

## Architecture

### Package layout

- **`airsenal/framework/`** — all core logic; statistical models, database schema, optimization, squad/player classes, data fetching
- **`airsenal/scripts/`** — CLI entry points; ideally just parse args and call framework functions
- **`airsenal/tests/`** — pytest tests for framework code
- **`airsenal/data/`** — static historical FPL data (multiple seasons, used to seed the database)
- **`airsenal/api/`** — optional Flask API (work in progress)
- **`airsenal/scraper/`** — web scraping utilities (e.g., Transfermarkt)

### Data flow

1. **Database init** (`fill_db_init.py`) — loads historical season data from `airsenal/data/` into a local SQLite database
2. **Database update** (`update_db.py`) — fetches current-season fixtures, results, and player attributes from the FPL API via `curl_cffi`
3. **Prediction** (`fill_predictedscore_table.py`) — runs BPL (Bayesian Premier League) team models and player-level models to predict points; writes to `PlayerPrediction` table
4. **Optimization** (`fill_transfersuggestion_table.py`) — uses a greedy/brute-force search to find optimal transfers; writes to `TransferSuggestion` table
5. **Apply** (`make_transfers.py`, `set_lineup.py`) — optionally posts transfers and lineup to the FPL API. NEVER run `make_transfers.py` yourself whilst testing changes as this leads to irreversible changes to the actual AIrsenal FPL team entry.

`airsenal_run_pipeline` is the top-level orchestrator for steps 1–5.

### Key framework modules

| File | Purpose |
|------|---------|
| `schema.py` | SQLAlchemy ORM models (`Player`, `Fixture`, `PlayerScore`, `PlayerPrediction`, `Squad`, etc.) |
| `data_fetcher.py` | FPL API client (uses `curl_cffi`); handles auth and data fetching |
| `prediction_utils.py` | BPL team-level match score predictions |
| `player_model.py` | Conjugate Bayesian and Numpyro player performance models |
| `optimization_utils.py` | Transfer optimization logic (greedy/brute-force) |
| `optimization_squad.py` | Initial whole-squad optimization (DEAP genetic algorithm) |
| `squad.py` | `Squad` class: 15 players, formation/budget constraint checking |
| `transaction_utils.py` | Transfer transaction management |
| `utils.py` | Shared utilities and default database session |

### Database

SQLite, default location: `$AIRSENAL_HOME/data.db` (configurable via `AIRSENAL_DB_FILE` env var). SQLAlchemy v2.0+ ORM. The `dbsession` argument (defaulting to the session created in `schema.py`) is threaded through most framework functions.

### Configuration

Required env var: `FPL_TEAM_ID`. Optional: `FPL_LOGIN`, `FPL_PASSWORD`, `FPL_LEAGUE_ID`, `AIRSENAL_DB_FILE`. Use `airsenal_env set` to persist these under `AIRSENAL_HOME`.

## Code conventions

- **Branch naming:** `feature/<issue>-<description>` or `bugfix/<issue>-<description>`; all new branches should be made from `develop`, and all pull requests should be made to merge into `develop`
- **Function argument order** (where applicable): other args → `player`/`player_id` → `position` → `team` → `tag` → `gameweek` → `season` → `fpl_team_id` → `dbsession` → `apifetcher` → `verbose`
- **Season strings:** `"2122"` for the 2021/22 season
- **Position strings:** `"GK"`, `"DEF"`, `"MID"`, `"FWD"`, or `"all"`
- Docstrings should follow numpydoc convention; type hints are encouraged
