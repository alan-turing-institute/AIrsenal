# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is AIrsenal

AIrsenal is a machine learning package for optimizing Fantasy Premier League (FPL) team
selection and transfer decisions. It uses Bayesian statistical models to predict
player/team performance, a greedy/brute-force approach to optimize transfers, and a DEAP
genetic algorithm for initial whole-squad selection — all under FPL constraints (budget,
squad size, position limits, chips, etc.).

## Read these first

The durable documentation is in the repository, written for anyone working here rather
than for an agent. Prefer it to anything restated below:

- **[docs/architecture.md](docs/architecture.md)** — the package chain, what each package
  owns, which contracts enforce it, and a file-by-file map.
- **[docs/adding-a-model.md](docs/adding-a-model.md)** — the five pluggable component
  kinds, a worked example of adding one, and how to find out whether it is any better.
- **[docs/how-it-works.md](docs/how-it-works.md)** — the database schema and how points
  predictions are built.
- **[CodingConventions.md](CodingConventions.md)** — where code goes, branch naming,
  argument order, docstring style.

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

Offline is enforced (`--disable-socket`), and `slow` and `live` tests are deselected by
default. Coverage has a floor: see `[tool.coverage.report]` in `pyproject.toml`.

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

All of these also run as pre-commit hooks, so this is mostly for running them directly.
`mypy` checks `src/airsenal` and `tools`; `lint-imports` always checks the whole package,
because a layering violation is an edge between two modules and there is nothing to
narrow to.

**Pre-commit hooks:**
```bash
pre-commit install
pre-commit run --all-files
```

**Run the full pipeline (typical usage):**
```bash
uv run airsenal run
```

## Data flow

1. **Database init** (`ingest/init_db.py`) — loads historical season data from
   `src/airsenal/data/` into a local SQLite database
2. **Database update** (`ingest/update.py`) — fetches current-season fixtures, results,
   and player attributes from the FPL API via `curl_cffi`
3. **Prediction** (`prediction/run.py`) — runs BPL team models and player-level models to
   predict points; writes to `PlayerPrediction` table
4. **Optimization** (`optimization/run_transfers.py`) — searches for optimal transfers;
   writes to `TransferSuggestion` table
5. **Apply** (`apply/transfers.py`, `apply/lineup.py`) — optionally posts transfers and
   lineup to the FPL API

`airsenal run` is the top-level orchestrator for steps 1-5.

## Rules for working here

**Never run `airsenal apply`, `make_transfers` or `set_lineup` while testing changes.**
They write irreversibly to the real AIrsenal FPL entry. Use `--dry-run`, which builds the
payload and posts nothing, or assert on what `build_transfer_payload` returns.

**Prediction is single-threaded by design.** Don't add multi-threading or multiprocessing
to `prediction/run.py`, or to any code that calls a jax-based model: jax deadlocks under
multi-threading. Prediction is fast enough without it.

**The transfer search must fork,** and can only fork before jax has been initialised — see
`core/concurrency.py`.

**Adding a model is a table entry, not a special case.** If adding one seems to need
edits anywhere but the class and its table line, the seam is in the wrong place. See
[docs/adding-a-model.md](docs/adding-a-model.md).

## Conventions worth knowing before you edit

Full versions in [CodingConventions.md](CodingConventions.md). The machine-checked ones:

- **Positions and chips:** use the `Position` and `Chip` enums from `game/enums.py`, not
  bare strings (`"all"` is still a plain string where a position filter accepts it).
  Enforced by `tests/test_naming_conventions.py`.
- **Gameweek naming:** `gameweek`, `gameweeks`, `n_gameweeks`. Same test.
- **Argument order:** `other args → player/player_id → position → team → tag → gameweek →
  season → fpl_team_id → dbsession → fetcher → verbose`. `tests/test_argument_order.py`
  enforces it as a ratchet — 53 functions predate the check and are listed there, and
  nothing may be added to that list.
- **Notebook imports** must resolve against the package: `tests/test_notebooks.py`.
- **Docstrings:** [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings),
  usually one line, and the first line is a summary and only that. An
  `Args:`/`Returns:`/`Raises:` section is for what the signature does not already say — a
  unit, a sentinel value's meaning, a side effect — not a restatement of it.
- Document what the code does now. Rationale belongs in a docstring only when it
  constrains future work; why the code changed belongs in the commit message.
- **Season strings:** `"2122"` for the 2021/22 season.
- **Branch naming:** `feature/<issue>-<description>` or `bugfix/<issue>-<description>`,
  from `develop`, and pull requests merge into `develop`.
