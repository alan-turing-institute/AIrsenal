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

## Code style, formatting, code quality

We are generally following the [PEP-8 style guide](https://www.python.org/dev/peps/pep-0008/) regarding conventions for class, function, and variable names.

For code formatting, we use the `black` linter before pushing our changes to Github - this can be run from the main "AIrsenal" directory by doing:
```
black .
```
and it will reformat any python files it finds.

We furthermore use the "flake8" style checker - this will flag up any unused imports, or undefined variables, for you to fix.  This can be run, again from the main AIrsenal directory, via:
```
flake8
```

## Order of function arguments

Many functions in AIrsenal take a large number of arguments.  Where possible, it
would be good to standardise the order in which these arguments go across different functions.  This is currently not enforced, and is complicated by different arguments having or not having default values (which would favour putting them towards the end) in different functions, but where possible, we could try to move towards a common order.

Below is a suggested ordering of commonly used arguments, from first to last:
* *player* or *player_id* (instance of Player class, or the player_id in the database for that player)
* *position* (str, either "GK", "DEF", "MID" or "FWD", or "all")
* *team* (str, 3-letter identifier for team, e.g. "ARS, MUN", or "all")
* *tag* (str, a unique identifier for a set of entries (e.g. points predictions) in the database)
* *gameweek*, or *gameweek_range* (may have default value NEXT_GAMEWEEK)
* *season* (often has a default value "CURRENT_SEASON")
* *fpl_team_id* (the ID of the squad in the FPL API - can be seen on the FPL website by looking at the URL after clicking on "View gameweek history").
* *dbsession* (database session - often has default value "session", which is the default session created in `schema.py`.)
* *apifetcher* (instance of FPLDataFetcher, often has default value "fetcher", which is the default instance created in `utils.py`)
* *verbose* (boolean, if True, print out extra information)