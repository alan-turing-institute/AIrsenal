# Code conventions for AIrsenal

This note aims to capture some of the practices we have (consciously or
unconsciously) adopted during the development of AIrsenal, with a view
to making the code more consistent, and therefore easier to develop.

It is not intended as a set of hard-and-fast rules - there will always be
exceptions, and we definitely don't want to deter anyone from contributing,
rather we hope that this will develop into a set of helpful guidelines, and
additions/corrections to this document are always welcome!

## Git branches

In the original AIrsenal repo in the "alan-turing-institute" organization, we
maintain two long-running branches: "main" and "develop".
The aim is that "main" will always be functional, and this is likely to be the
first entrypoint to the project for new users.
The "develop" branch should hopefully also be functional, but it is not inconceivable that merging a new feature could break it for brief periods.

Development should mostly take place on individual branches that are branched off "main" or "develop".   As a general guideline, new features should be branched off, and merged back into "develop", while bugfixes can be branched off, and merged back into "main" (and then merge "main" into "develop").
The naming convention for branches should generally be something like
`feature/<github_issue_number>-<brief-description>` or `bugfix/<github_issue_number>-<brief-description>` (and as this implies, there should ideally be a corresponding Issue!).

## Developer Dependencies

Packages used for developing AIrsenal but not needed to run AIrsenal (such as those in the code style and formatting section below) are included in the `dev` optional dependencies group. To install them run the following command from the `AIrsenal` directory:
```
uv sync --extra dev
```

## Code style, formatting, code quality

We are generally following the [PEP-8 style guide][link_pep8] regarding conventions for class, function, and variable names.

Ideally, docstrings should follow [numpydoc][link_numpydoc] convention (though this is not always the case in the existing code).
We encourage extensive documentation.

We use type hints, as provided by the [typing](link_typing) module, and `mypy` runs in
strict mode in CI over the whole of `src/airsenal`, with no per-module exemptions from
annotating (`uv run mypy` to run it yourself). The only `[[tool.mypy.overrides]]` blocks left in `pyproject.toml` name the
untyped third-party libraries and the handful of modules allowed to call into them
(jax/numpyro/bpl, deap, curl_cffi); everything else reaches those through our own
typed wrappers. Do not add a module to those blocks to get a new error to go away.

For code formatting and linting, we use [ruff](https://docs.astral.sh/ruff/) which combines the functionality of black, isort, and flake8. This can be run from the main "AIrsenal" directory by doing:
```
ruff check --fix .  # Linting with automatic fixes
ruff format .       # Code formatting
```

Finally, we have a [pre-commit](https://pre-commit.com/) setup that runs the formatting and linting above whenever you commit. It deliberately stops there: `mypy` and `lint-imports` both analyse the whole package rather than the files you happened to touch, so they run in CI instead, and are worth running yourself before pushing. To set pre-commit up run this from the AIrsenal directory:
```
pre-commit install
```
To check they're working run:
```
pre-commit run --all-files
```

## Where to put the code

The package that gets built lives in `src/airsenal`. Its subdirectories form a one-way
dependency chain, from the most general at the bottom to the most specific at the top:

```
cli            command definitions and argument parsing, nothing else
pipeline       orchestration: `run` and `replay`
apply          the only code that writes to the real FPL entry
optimization   the transfer search and the whole-squad builder
ingest         export
reporting      squad           prediction
db             the tables, the queries and the session
remote         everything that talks to the internet, and nothing else
core           no airsenal-specific dependencies at all
```

A module may import from the rows below it, never from the rows above. This is checked
in CI, and worth running yourself - `uv run lint-imports` - because a single
convenience import in the wrong direction is what turned an earlier version of the
codebase into one module that everything depended on, and it does not fail at runtime,
so nothing catches it until something is looked for.

Three rules decide where new code goes:

1. **If it imports no other airsenal module, it goes in `core/`.** That covers both FPL's
   own rules (what a goal is worth, what a position is, how season strings work) and
   generic plumbing (dates, logging, the console, caching).
2. **Otherwise it belongs to the stage that owns it** - the list above. If it seems to
   belong to two stages, it goes in the lower one, or it is two functions.
3. **A new subdirectory needs at least three modules.** A directory holding one file
   tells the reader that a category exists without telling them what is in it.

`src/airsenal/data` holds the packaged historical CSV and JSON that seeds the database.
Resolve paths into it with `airsenal.core.data_files.data_file()`, never by joining onto
`__file__` - that only works while the calling module sits at one particular depth.

`tests/` sits at the repository root, outside the package, and mirrors the package where
there is enough to mirror. When adding new functionality it is always a good idea to
write a corresponding test, and to run the full suite to check nothing else broke:
`uv run pytest tests`.

There is also a `notebooks` directory, which contains Jupyter notebooks used to develop,
test or demonstrate various bits of AIrsenal functionality. These can be a good starting
point to experiment and familiarize yourself with the code. `tools/` holds dev one-offs
that are not packaged; install them with the `tools` extra.

## Order of function arguments

Many functions in AIrsenal take a large number of arguments.  Where possible, it
would be good to standardise the order in which these arguments go across different functions.  This is currently not enforced, and is complicated by different arguments having or not having default values (which would favour putting them towards the end) in different functions, but where possible, we could try to move towards a common order.

Below is a suggested ordering of commonly used arguments, from first to last:
* Other arguments (not listed below)
* *player* or *player_id* (instance of Player class, or the player_id in the database for that player)
* *position* (str, either "GK", "DEF", "MID" or "FWD", or "all")
* *team* (str, 3-letter identifier for team, e.g. "ARS, MUN", or "all")
* *tag* (str, a unique identifier for a set of entries (e.g. points predictions) in the database)
* *gameweek*, or *gameweek_range* (int, or list of ints, may have default value NEXT_GAMEWEEK)
* *season* (str, e.g. "2122" for the 2021/2022 season, often has a default value "CURRENT_SEASON")
* *fpl_team_id* (str, the ID of the squad in the FPL API - can be seen on the FPL website by looking at the URL after clicking on "View gameweek history").
* *dbsession* (database session - often has default value "session", which is the default session created in `schema.py`.)
* *apifetcher* (instance of FPLDataFetcher, often has default value "fetcher", which is the default instance created in `utils.py`)
* *verbose* (boolean, if True, print out extra information)

[link_numpydoc]: https://numpydoc.readthedocs.io/en/latest/format.html
[link_pep8]: https://www.python.org/dev/peps/pep-0008/
[link_typing]: https://docs.python.org/3/library/typing.html
