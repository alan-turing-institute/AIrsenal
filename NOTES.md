Note that running any of the commands mentioned below assume you have AIrsenal installed and its environment activated (if it was installed in a virtual environment).

## Database

The database is filled with data from the previous three FPL seasons (stored in the repo at `airsenal/data`), as well as data from the current season (obtained from the FPL API). It contains the following tables:

**_Tables with Football Club Data:_**
- **Team:**

- **Fixture:** 

- **Result:**

- **FifaTeamRating:**

  
**_Tables with Player Data:_**
- **Player:** Name and ID for each player (for all football players that have been in the FPL game at some point in the last 3 seasons). The ID is the same as the FPL player ID for players active in the current season.

- **PlayerAttributes:** Attributes for each player in every gameweek of each season. The main attributes are FPL price, FPL position, and the team the player plays for. 

- **PlayerScore:**

**_Tables with FPL Squad Data:_**

- **Transsction:**

**_Tables with AIrsenal Data:_**

- **PlayerPrediction:**

- **TransferSuggestion:**

The database also contains information about the user's current FPL squad and AIrsenal's point predictions for each player and transfer recommendations

The database schema is defined using `sqlalchemy` in `airsenal.framework.schema.py`.

### Initial Database Setup
To setup the AIrsenal database:
1. Remove any pre-existing database:
  `rm /tmp/data.db`
2. Run  `airsenal_setup_initial_db`
(previously `setup_airsenal_database`)
3. A sqlite3 database will be created at `/tmp/data.db` containing the AIrsenal data.

This runs the function `main()` in `airsenal.scripts.fill_db_init`, which calls functions in other scripts to fill the individual tables from the data files and API.

### Updating the Database


## Player Points Predictions

## Transfer Optimisation

---
# Old notes

## Data wrangling

Different aspects of the data come from different sources:

 * Player data for the upcoming 2018/19 season, and from the 2017/18 season, are from the FPL API.
 * Player data from previous seasons are from scraping the FPL archives page.
 * Match results from previous seasons are copy/pasted from Sky Sports website, and parsed.
 * Fixtures for 2018/19 are copy/pasted from the FPL website and parsed.

Many of these different sources can have slightly different names for players and/or teams (e.g. FPL
insists on using "Spurs" for some reason).  There are a couple of very naive scripts:
```match_team_names.py``` and ```match_player_names.py``` which aim to reconcile some of these
differences.

For both team names and player names, we will use the FPL values ("first_name second_name" for players).
The scripts mentioned above will generate dictionaries with these names as the keys, and with a list
of possible alternative names as the values.  These dictionaries (saved in json files) will be used
when writing into the SQLite database.

The "results_xxyy.csv" files contain the date of matches, but not their
corresponding "gameweek" in the FPL season (maybe we don't need this anyway,
but it could be useful to match up players with historic results.  There is
a script ```find_gameweek_for_match.py``` that attempts to match these up
using the player_detail_xxyy.json data, but it is not perfect, and the results
require manual tweaking. (To be investigated).

## Database

The proposed schema of the database is as follows:

```
Player Table
============
player_id, player_name, current_team, current_price, purchased_price
```
(The purchased_price is necessary to calculate how much the player can be
sold for)

```
Match Table
===========
match_id, date, season, gameweek, home_team, away_team, home_score, away_score
```

```
Fixture Table
=============
fixture_id, date, gameweek, home_team, away_team
```

````
PlayerScore Table
=================
player_id, match_id, played_for, opponent, score, ... , ...
```
(Note that this has columns for played_for and opponent since players can
change teams)

```
PlayerPrediction Table
======================
player_id, gameweek, predicted_score, method
```
(the method column allows us to have more than one predicted score per
player per gameweek, and add more later, without having to add more columns.
We now have a gameweek column rather than a fixture_id column - in double gameweeks
the player can have more than one fixture in a gameweek - the score we show is the 
sum of both.)

```
Transaction Table
=================
player_id, gameweek, bought_or_sold
```
(the bought_or_sold column can have values +1 for buying a player or -1 for selling)
