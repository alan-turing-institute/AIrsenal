# AIrsenal
[![Build Status](https://travis-ci.org/alan-turing-institute/AIrsenal.svg?branch=master)](https://travis-ci.org/alan-turing-institute/AIrsenal)

Machine learning Fantasy Premier League team

## Install

To install, you will need a working C++ compiler and `pip` (>=18.0).
If you're using anaconda, run `conda install gcc` in the same environment.
Install the `airsenal` module from source via

```bash
pip install https://github.com/alan-turing-institute/AIrsenal/archive/master.zip --process-dependency-links
```

## Getting started

Once you've installed the module, you will need to set three environment variables:

1. `FD_API_KEY`: an API key for [football data](https://www.football-data.org/)
2. `FPL_TEAM_ID`: the team ID for your FPL side.
3. `FPL_LEAGUE_ID`: a league ID for FPL (this is only required for a small subset of functionality).

Once this is done, run the following command

```shell
setup_airsenal_database
```

You should get a file ```/tmp/data.db```.  This will fill the database with all that is needed up to the present day.

## Updating

TODO: update this once entry points are finished.

To stay up to date in the future, you will need to fill three tables: ```match```, ```player_score```, and ```transaction```
with more recent data, using a couple of scripts as detailed below.

To fill the match results, we will query the football-data.org API.  You need an API key for this (sign up at https://www.football-data.org/ ) - put it into a file ```data/FD_API_KEY```
Then run (from ```scripts```)
```
python fill_match_table.py --input_type api --gw_start <first_gameweek> --gw_end <last_gameweek+1>
```


Once the match data is there, you can fill the player score data by running (also from ```scripts```)
```
python python fill_playerscore_this_season.py --gw_start <first_gameweek> --gw_end <last_gameweek+1>
```

The transaction table is a bit different, as this reflects the players we are buying and selling in our own team.
As such, you will need to edit ```scripts/fill_transaction_table.py``` yourself to fill in the player_ids of players
transferred in or out, and then run the script with ```python fill_transaction_table.py```.
