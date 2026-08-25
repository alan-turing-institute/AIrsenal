# Code conventions for AIrsenal

Guidelines, not hard-and-fast rules — but following them keeps the code consistent, and
additions and corrections are welcome.

## Git branches

The `alan-turing-institute/AIrsenal` repo has two long-running branches. `main` is always
functional and is where new users should start. `develop` should also be functional, but
merging a new feature could break it briefly.

Branch off `develop` and open pull requests against `develop`. Name branches
`feature/<github_issue_number>-<brief-description>` or
`bugfix/<github_issue_number>-<brief-description>` — which implies there should be a
corresponding issue.

## Developer dependencies

Everything needed to develop but not to run AIrsenal is in the `dev` optional dependency
group:
```
uv sync --extra dev
```

## Code style, formatting, code quality

We follow [PEP-8][link_pep8] for class, function and variable names, and
[Google style][link_google_docstrings] for docstrings.

Most docstrings should be one line. Add an `Args:` or `Returns:` section only for
what a reader cannot get from the name, the type hint and the default - a unit
(`budget` is in tenths of a million), a value the name does not imply (`None` means
bench boost), a side effect, or which exception escapes. Leave the rest out: a
parameter list that restates the signature is what rots first, and every entry in it
is one more thing to keep true.

We use type hints, and `mypy` runs in strict mode over the whole of `src/airsenal` with
no per-module exemptions from annotating (`uv run mypy`). The `[[tool.mypy.overrides]]`
blocks in `pyproject.toml` name the untyped third-party libraries and the handful of
modules allowed to call into them directly; everything else reaches those through our own
typed wrappers. Do not add a module to those blocks to make a new error go away.

For formatting and linting we use [ruff](https://docs.astral.sh/ruff/), which covers what
black, isort and flake8 used to:
```
ruff check --fix .  # Linting with automatic fixes
ruff format .       # Code formatting
```

A [pre-commit](https://pre-commit.com/) config runs all of the above on every commit —
formatting, linting, type checking, and the layering contracts — and the same hooks run
over every file in CI:
```
pre-commit install
pre-commit run --all-files
```

## Where to put the code

The package lives in `src/airsenal`. Its subdirectories form a one-way dependency chain,
most general at the bottom:

```
cli            command definitions and argument parsing, nothing else
pipeline       orchestration: `run` and `replay`
apply          the only code that writes to the real FPL entry
optimization   the transfer search and the whole-squad builder
export         writing data back out
ingest         filling the database from packaged data and the FPL API
reporting      rendering results: tables, plots, Discord posts
squad          the Squad class and the state of the user's own entry
prediction     the models, and the points they imply
db             the tables, the queries and the session
remote         everything that talks to the internet, and nothing else
core           generic plumbing: logging, caching, the console, date parsing
game           the facts about FPL - it imports nothing at all
```

A module may import from the rows below it, never from the rows above. That is what the
import-linter contract in `pyproject.toml` enforces, and it is `exhaustive`, so a new
package has to be given a place in the chain rather than silently escaping it. The
pre-commit hook checks it, but run `uv run lint-imports` yourself after moving code
between packages: a wrong-direction import does not fail at runtime, so nothing else
catches it.

Three questions decide where new code goes:

1. **Is it a fact about Fantasy Premier League?** What a goal is worth, what a position
   is, how season strings work, what other data sources call a club — that goes in
   `game/`, which imports nothing at all: no airsenal package, no third-party library,
   not even a logger. Needing a logger or a dataframe is the signal that it is not a fact
   about the game.
2. **Is it generic Python machinery with no airsenal imports?** That goes in `core/`.
3. **Otherwise it belongs to the stage that owns it** — the list above. If it seems to
   belong to two stages, it goes in the lower one, or it is two functions.

Prefer a module in an existing package to a new subdirectory holding one file: a
directory tells the reader a category exists without telling them what is in it.

`src/airsenal/data` holds the packaged historical CSV and JSON that seeds the database.
Resolve paths into it with `airsenal.core.data_files.data_file()`, never by joining onto
`__file__` — that only works while the calling module sits at one particular depth.

`tests/` sits at the repository root, outside the package, and mirrors the package where
there is enough to mirror. When adding functionality, add a test, and run the whole suite
to check nothing else broke: `uv run pytest tests`.

`notebooks/` holds Jupyter notebooks used to develop, test or demonstrate bits of
AIrsenal, and can be a good place to start experimenting. `tools/` holds dev one-offs
that are not packaged; install them with the `tools` extra.

For how the pieces fit together at runtime — the database schema and how points
predictions are built — see [docs/how-it-works.md](docs/how-it-works.md).

## Order of function arguments

Many AIrsenal functions take a lot of arguments. Where possible, order them like this:

* Other arguments (not listed below)
* *player* or *player_id* (instance of Player class, or the player_id in the database for that player)
* *position* (`Position` from `game/enums.py`; `"all"` is still a plain string where a position filter accepts one)
* *team* (str, 3-letter identifier for team, e.g. "ARS, MUN", or "all")
* *tag* (str, a unique identifier for a set of entries (e.g. points predictions) in the database)
* *gameweek* (int - one gameweek), or *gameweeks* (list of ints). A count is *n_gameweeks*. `tests/test_naming_conventions.py` enforces those three names, and that a parameter called `gameweek` is not secretly a list.
* *season* (str, e.g. "2122" for the 2021/2022 season, often has a default value "CURRENT_SEASON")
* *fpl_team_id* (str, the ID of the squad in the FPL API - can be seen on the FPL website by looking at the URL after clicking on "View gameweek history").
* *dbsession* (database session - usually defaulting to None and resolved with `get_session()` from `db/session.py`)
* *fetcher* (instance of FPLDataFetcher - usually defaulting to None and resolved with `get_fetcher()` from `remote/fpl_api.py`)
* *verbose* (boolean, if True, print out extra information)

[link_google_docstrings]: https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
[link_pep8]: https://www.python.org/dev/peps/pep-0008/
