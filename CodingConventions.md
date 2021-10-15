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

Packages used for developing AIrsenal but not needed to run AIrsenal (such as those in the code style and formatting section below) are included in
`requirements-dev.txt`. To install them run the following command from the `AIrsenal` directory (with your AIrsenal virtual environment activated if you are using one, for example with `conda activate airsenalenv`):
```
pip install -r requirements-dev.txt
```

## Code style, formatting, code quality

We are generally following the [PEP-8 style guide][link_pep8] regarding conventions for class, function, and variable names.

Ideally, docstrings should follow [numpydoc][link_numpydoc] convention (though this is not always the case in the existing code).
We encourage extensive documentation.

Although there are not many in the current codebase, we also encourage the use of type hints, as provided by the [typing](link_typing) module. Examples of functions using this can be found in `airsenal/framework/player_model.py`.

For code formatting, we use the `black` linter before pushing our changes to Github - this can be run from the main "AIrsenal" directory by doing:
```
black .
```
and it will reformat any python files it finds.

We also use [isort](https://pycqa.github.io/isort/index.html) to have a consistent alphabetical order on imports. This can be run from the "AIrsenal" directory with:
```
isort .
```

We furthermore use the "flake8" style checker - this will flag up any unused imports, or undefined variables, for you to fix.  This can be run, again from the main AIrsenal directory, via:
```
flake8
```

Finally, we have a [pre-commit](https://pre-commit.com/) setup that you can use to run all the steps above whenever commiting something to the AIrsenal repo. To set these up run this from the AIrsenal directory:
```
pre-commit install
```
To check they're working run:
```
pre-commit run --all-files
```

## Where to put the code

Within the main `AIrsenal` directory, the python package that gets built is based on the `airsenal` subdirectory.   In this subdirectory, there are three further important subdirectories: `tests`, `framework`, and `scripts`.
* *framework* is where the vast majority of code should live.  Things like the player-level statistical model and the database schema can be found here, as well as class definitions for e.g. "Squad" and "CandidatePlayer", the "DataFetcher" that retrieves information from the API, and several modules containing utility functions.
* *tests* contains test code for checking the behaviour of the functions in *framework*.  When adding new functionality, it is always a good idea to write a corresponding test, and to run the full suite of tests to ensure that existing functions aren't broken. To check all tests are passing run `pytest` from the AIrsenal directory.
* *scripts* contains the command-line interfaces for running the various steps of AIrsenal (initial database setup, database update, prediction, and optimization).  Ideally these scripts would just parse command-line arguments and then call library functions from `framework`, but in practice some of them do contain more logic than that.

There is also a `notebooks` directory in the main `AIrsenal` directory, which contains Jupyter notebooks that have been used to develop, test, or demonstrate, various bits of AIrsenal functionality.   These can be a good starting point to experiment, and familiarize yourself with the code.


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
