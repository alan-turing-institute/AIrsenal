# AIrsenal
[![Build Status](https://travis-ci.org/alan-turing-institute/AIrsenal.svg?branch=master)](https://travis-ci.org/alan-turing-institute/AIrsenal)

Machine learning Fantasy Premier League team

## How to run

After cloning this repo, you can start to populate an sqlite file with historical data with:
```
cd scripts
source fill_db.sh
```

You should get a file ```/tmp/data.db```.  This will fill the database with everything up to the present date.


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
transfered in or out, and then run the script with ```python fill_transaction_table.py```.