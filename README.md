# AIrsenal
Machine learning Fantasy Premier League team

## How to run

After cloning this repo, you can start to populate an sqlite file with historical data with:
```
cd scripts
source fill_db.sh
```

You should get a file ```/tmp/data.db```.  This will (as of 6/9/18) contain data up to gameweek 3 of the 18/19 season.
To stay up to date, you need to fill two tables ```match``` and ```player_score``` with more recent data, using a couple of scripts as detailed below.

To fill the match results, we will query the football-data.org API.  You need an API key for this (sign up at https://www.football-data.org/ ) - put it into a file ```data/FD_API_KEY```
Then run (from ```scripts```)
```
python fill_match_table.py --input_type api --gw_start <first_gameweek> --gw_end <last_gameweek+1>
```


Once the match data is there, you can fill the player score data by running (also from ```scripts```)
```
python python fill_playerscore_this_season.py --gw_start <first_gameweek> --gw_end <last_gameweek+1>
```






