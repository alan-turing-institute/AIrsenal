# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AIrsenal is a machine learning package for optimizing Fantasy Premier League (FPL) team selections. It uses statistical models to predict player performance and suggests optimal transfers and squad formations.

**Python Support:** 3.10+ (`.python-version` pins 3.14; jaxlib is no longer version-pinned)

## Environment

This repo lives under `~/Documents/Claude/`, which is iCloud-synced. **The venv must live
outside iCloud** — with `.venv` in-tree, iCloud materialises files lazily and `import
pandas` alone can block for minutes, which looks like a hang rather than an error.

Set this in your shell profile (already required for every command below):
```bash
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/airsenal"
```

## Common Commands

All commands need `uv run --no-sync` (the `--no-sync` avoids a reinstall on every
invocation, which is slow even with the venv outside iCloud).

### Setup & Installation
```bash
uv sync --all-extras       # Install everything; needed for the full test suite
```
`--all-extras` rather than `--extra dev`: `test_api_utils.py` imports Flask, which lives
in the `api` extra, so a dev-only install fails at collection.

### Testing
```bash
uv run --no-sync pytest airsenal/tests                # Run all tests (~60s)
uv run --no-sync pytest airsenal/tests/test_squad.py  # Run single test file
uv run --no-sync pytest airsenal/tests -k "test_name" # Run tests matching pattern
```
Without `uv run`, a bare `pytest` may resolve to the system Python and fail with a
confusing `PackageNotFoundError: No package metadata was found for airsenal`.

### Code Quality
```bash
ruff check --fix .         # Lint with automatic fixes
ruff format .              # Format code
pre-commit run --all-files # Run all pre-commit hooks
pre-commit install --install-hooks  # Setup git hooks (first time)
```

### AIrsenal Pipeline
```bash
airsenal_run_pipeline      # Run full pipeline (setup/update, predict, optimize)
airsenal_setup_initial_db  # Initialize database with 3 seasons of data
airsenal_update_db         # Fetch latest data from FPL API
airsenal_run_prediction --weeks_ahead 3   # Generate player predictions
airsenal_run_optimization --weeks_ahead 3 # Suggest transfers
airsenal_report            # Summarise squad, XI, captain and transfers in one place
airsenal_env get           # View configuration
airsenal_env set -k KEY -v VALUE  # Set config value
```

### Model and optimiser tuning

These change the advice you get. **Three default to off because the right value is
empirical — settle them with `airsenal_replay_season` on a finished season, not by
eye.** Turning them on untested is a guess dressed up as a setting.

| Flag | Command | Default | What it does |
|---|---|---|---|
| `--xg_weight` | prediction | 0 | Blend expected goals/assists into the player model (0-1) |
| `--min_hit_gain` | optimisation | 0 | Points a hit must gain over the best no-hit strategy |
| `--condition_bonus_on_fixture` | prediction | off | Scale bonus by how favourable the fixture is |
| `--discount` | optimisation | 14/15 | Per-gameweek discount: the dial between this week and next month |
| `--num_candidates` | optimisation | 5 | Affordable replacements scored per player considered for sale |
| `--min_fixtures_behind` | prediction | 3 | Recent matches used to estimate minutes |
| `--seed` | pipeline | none | Reproducible suggestions; without it the search varies run to run |
| `--check_data` | pipeline | off | Run the sanity checks after the DB update |
| `--consider_available_chips` | pipeline | off | Consider any chip the API says you still hold |

## Architecture

### Directory Structure
- `airsenal/framework/` - Core business logic (models, optimization, utilities)
- `airsenal/scripts/` - CLI entry points that call framework functions
- `airsenal/tests/` - Test suite with fixtures in `conftest.py`
- `airsenal/data/` - Historical JSON data files
- `notebooks/` - Jupyter notebooks for experimentation

### Key Modules
| Module | Purpose |
|--------|---------|
| `framework/schema.py` | SQLAlchemy ORM models (Player, Fixture, PlayerPrediction, etc.) |
| `framework/data_fetcher.py` | FPL API client using curl_cffi |
| `framework/squad.py` | Squad validation and management |
| `framework/optimization_squad.py` | Genetic algorithm for full squad building |
| `framework/optimization_transfers.py` | Genetic algorithm for transfer strategy |
| `framework/player_model.py` | ML models for player predictions (JAX-based) |
| `framework/prediction_utils.py` | Points calculation logic |
| `framework/utils.py` | Heavily-used utility functions |
| `framework/env.py` | Cross-platform configuration management |

### Data Flow
1. `data_fetcher.py` pulls data from FPL API
2. Data stored in SQLite database (schema in `schema.py`)
3. `player_model.py` + `bpl_interface.py` generate predictions
4. `optimization_*.py` uses DEAP genetic algorithms to find optimal transfers
5. Results stored as `TransferSuggestion` and `PlayerPrediction` in database

## Code Style

- **PEP-8** with 88 character line length (ruff)
- **Docstrings:** NumPy format preferred
- **Type hints:** Encouraged, see `player_model.py` for examples
- **MyPy:** Runs on `airsenal/framework` and `airsenal/scripts` only

### Function Argument Order Convention
When functions take many arguments, follow this order:
1. Other args
2. player/player_id
3. position
4. team
5. tag
6. gameweek/gameweek_range
7. season
8. fpl_team_id
9. dbsession
10. apifetcher
11. verbose

## Git Workflow

**Remotes matter here.** `origin` is the upstream project
(`alan-turing-institute/AIrsenal`); `barrytho` is this fork and is the one to pull and
push. `git pull origin main` drags in upstream work this setup has not been tested
against — a mistake already made once in the scheduled runner script.

- **main** - Always functional, user-facing. Topic branches are merged into it here.
- Features: branch from `main` as `feature/<description>`
- Bugfixes: branch from `main` as `bugfix/<description>`
- Do NOT rebase or rewrite history
- Never push without being asked

When several topic branches touch the same functions, expect conflicts that are
mechanical (parameters added side by side, not competing logic) — keep both sides.

## Environment Variables

- `FPL_TEAM_ID` (required) - Your FPL team ID
- `FPL_LOGIN` / `FPL_PASSWORD` (recommended) - For API authentication
- `FPL_LEAGUE_ID` (optional) - For league standings
- `AIRSENAL_DB_FILE` (optional) - Custom database path
- `AIRSENAL_HOME` (optional) - Override config directory

## Testing Notes

- Tests use temporary in-memory SQLite database (configured in `conftest.py`)
- Test data available at `airsenal/tests/testdata/testdata_1718_1819.db`
- Coverage configured with pytest-cov in `pyproject.toml`
- The test data predates FPL publishing xG (2022/23), so anything touching
  `expected_goals` must cope with the column being entirely null

## Gotchas

Each of these has cost real time at least once.

**The database is the expensive artefact.** It lives at
`~/Library/Application Support/airsenal/data.db` and is not in git. Rebuilding means
re-ingesting several seasons, so `airsenal_update_db` now copies it first and keeps
the last 3 backups alongside it. Don't run `--clean` casually.

**Nothing is interactive-safe.** AIrsenal prompts on stdin when it has no FPL login,
and again when a database update fails. With no terminal those raise `EOFError` and
kill the run, so anything headless (cron, CI, cloud) must pipe input:
`yes n | uv run airsenal_run_pipeline ...`.

**Between seasons the code looks broken but isn't.** `CURRENT_SEASON` is derived from
the date, so it flips in June. Until the new season's fixtures are in the database,
`NEXT_GAMEWEEK` falls back to the FPL API at *import* time — so a slow or failing API
shows up as a slow or failing import of `airsenal.framework.utils`, far from the
apparent cause.

**A stray clone inside `airsenal/` is poison.** One ended up at `airsenal/AIrsenal/`
(233 MB): being on the import path, pytest collected its tests twice and the linters
scanned 4,000 extra files. Both it and `.claude/worktrees/` are gitignored now.

**FPL entry IDs must be checked, not assumed.** `entry/<id>/` returning 404 while other
IDs return 200 means the ID is wrong or the team doesn't exist for this season. The
configured `6321674` currently 404s. Squad, bank and transfer history are all public
for a valid ID — only chips-remaining and applying transfers need a login.

## Scheduled cloud run

A claude.ai routine runs this daily at 17:00 UTC against the `barrytho` fork:
`https://claude.ai/code/routines/trig_018LGAAdmdu1VUr8R9GCaXjc`

It exits immediately unless the next FPL deadline is 12-36 hours away — that window is
24h wide and the job is daily, so exactly one run per gameweek proceeds. Cron can't
express "24h before the deadline" directly because deadlines move between Friday,
Saturday and Tuesday. It has no credentials by design and is read-only: it recommends,
it never submits transfers.
