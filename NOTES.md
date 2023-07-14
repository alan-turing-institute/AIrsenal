## Links

- Introduction to AIrsenal and its first season (2018/19): https://www.turing.ac.uk/news/airsenal
- AIrsenal's difficult start to its second season (2019/20): https://www.turing.ac.uk/news/airsenal-difficult-second-season

## Installation

Note that running any of the commands mentioned below assume you have AIrsenal installed and its environment activated (if it was installed in a virtual environment).

For installation instructions see the [README](https://github.com/alan-turing-institute/AIrsenal)

## Database

The database is filled with data from the previous three FPL seasons (stored in the repo at `airsenal/data`), as well as data from the current season (obtained from the FPL API). A lot of the historic FPL data has been compiled with the help of the [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League) repo. The database also contains information about the user's current FPL squad and AIrsenal's point predictions for each player and transfer recommendations. It has the following tables:

**_Tables with Football Club Data:_**
- **Team:** Name (short and full) and FPL ID for each team in each of the last three (and current) premier league seasons.

- **Fixture:** All past and future scheduled fixtures (home team, away team, date, gameweek and season).

- **Result:** Match result for each fixture (if available) - home goals and away goals.

- **FifaTeamRating:** FIFA (the game) team ratings for each team in each season. These are taken from https://www.fifaindex.com/teams/. Their main use is to give an idea of team strength for promoted teams (but we are considering their use in general - see [Issue #136](https://github.com/alan-turing-institute/AIrsenal/issues/136))


**_Tables with Player Data:_**
- **Player:** Name and ID for each player (for all football players that have been in the FPL game at some point in the last 3 seasons). The ID is the same as the FPL player ID for players active in the current season.

- **PlayerAttributes:** Attributes for each player in every gameweek of each season. The main attributes are FPL price, FPL position, and the team the player plays for.

- **PlayerScore:** Stats for each player in each match. These include points scored, goals scored and condeded, assists, bonus points, minutes played and a few others. Only contains the stats available from the FPL API, we don't currently have other stats (like xG).

**_Tables with FPL Squad Data:_**

- **Transaction:** Contains the history of each player you have bought and sold in your FPL team, and for what price (and which gameweek). This is used to determine your starting team and budget before running transfer optimisations.

**_Tables with AIrsenal Data:_**

- **PlayerPrediction:** Stores predicted points for each player in each fixture from an AIrsenal prediction run.

- **TransferSuggestion:** Stores recommended transfers from AIrsenal optimisation runs.

The database schema is defined using `sqlalchemy` in `airsenal.framework.schema.py`.

### Interacting with the FPL API

The `FPLDataFetcher` class in `airsenal.framework.data_fetcher` contains functions for retrieving data from the FPL API.

The main use of this is for database setup and updates, but it is also used elsewhere, for example for checking a player's current injury status in the prediction code.

### Initial Database Setup
To setup the AIrsenal database: (in a terminal)
1. Remove any pre-existing database:
  `rm /tmp/data.db`
2. Run  `airsenal_setup_initial_db` (previously `setup_airsenal_database`).

This runs the function `main()` in `airsenal.scripts.fill_db_init`, which calls functions in other scripts to fill the individual tables from the data files and API. A sqlite3 database will be created at `/tmp/data.db` containing the AIrsenal data.

Run the update script afterwards to add the latest status of your FPL team (transfers made etc.) to the database - see below.

### Updating the Database

To update the AIrsenal database run `airsenal_update_db` (previously `update_airsenal_database`) in a terminal. This calls the function `main()` in `airsenal.scripts.update_results_transactions_db`.

It does the following:
1. Update player attributes with their latest values.
2. Adds any new match results.
3. Adds any new player scoores.
4. Adds any new FPL transfers that have been made.

You should always run `airsenal_update_db --noattr` after an initial database setup to get the latest status of your FPL team (which is not done as part of `airsenal_setup_initial_db`). Giving the `--noattr` flag means player attributes will not be updated, which is not needed if you have just setup the database (as FPL player attributes can only change once per day).

Note we don't currently have a way to update the list of currently active _players_ (only their attributes). This unfortunately means that if a new player is added to the game the whole database needs to be recreated. It's therefore best to follow the initial database setup steps above at the start of every gameweek.

### Data Sanity Checks

Use `airsenal_check_data` (previously `check_airsenal_data`) from the command-line, which runs the `run_all_checks` function in `airsenal.scripts.data_sanity_checks`, performs the following sanity checks on the AIrsenal database:
- All seasons have 20 teams.
- Each season has 3 new teams (promoted teams).
- Each season has 380 fixtures.
- Players correctly assigned to one of the teams in each fixture.
- 11 to 14 playesr appear for each team in each fixture (NOTE: this fails for the end of the 1920 season when 5 substitutes were allowed).
- Sum of player goals (and own goals) matches the final match score for each team (currently one error due to there being two players called Danny Ward in the 1819 season, see issue #62).
- Number of assists less than or equal to number of goals.
- Goals conceded matches goals scored by opponent.


## Player Points Predictions

Player points predictions are generated from three main components:
1. A team-level model to predict final score probabilities for each match.
2. A player-level model to predict the probability that a player scores or assists each goal his team scores.
3. The number of minutes the player has played in recent matches and his current injury or suspension status.

**For more details on the modelling parts of AIrsenal see here: https://www.turing.ac.uk/news/airsenal**

### Team Model

BPL package (written by Angus, one of the original AIrsenal developers): https://github.com/anguswilliams91/bpl-next

### Player Model

NumPyro model definition: `airsenal/framework/player_model.py`

### How Predicted Points are Calculated

First:
- Fit the team and player models.
- Predict the probability of each number of goals scored and conceded for each team in each fixture in the gameweek to consider, using the team model.
- Get the number of minutes each player played in the last three fixtures, and their current injury and suspension status.

Then calculate the different points contributions as below:

**Recent Minutes and Appearance Points:**

The function `get_recent_minutes_for_player` in `airsenal.framework.utils` returns the number of minutes a player played in their last 3 matches (by default).

For each of the number of minutes played in the last 3 fixtures (e.g. if a player played 0 mins, 70 mins and 90 mins in the last 3 games, we make 3 predictions for the next match assuming he will play 0 mins, 70 mins or 90 mins. For both attacking and defending points (see below), the probability of scoring, assisting, or conceding a goal is weighted by the number of minutes (fraction of the match) the player is estimated to play.

If a player is marked as injured or suspended in the API (obtained with `is_injured_or_suspended()` in `airsenal.framework.prediction_utils`) with a 50% or less chance of playing in a fixture we assume he'll score 0pts.

As per the FPL appearance points rules, players score 0pts if a player doesn't play, 1pt for less 60 minutes, and 2pts for 60 minutes or more. See `get_appearance_points` in `airsenal.framework.FPL_scoring_rules`.

**Attacking Points (Goals Scored):**

The following is done in `get_attacking_points()` in `airsenal.framework.prediction_utils`:

- probability team scores that number of goals
- possible permutations of number of goals and assists for the player given the team scores tbat many goals
- probability of each permutation of goals and assists using trinomial player model (probability player scores, assists or not involved in a goal)
- FPL points of each permutation given number of points for a goal and assist for player's position.
- multiply probabilities and pts totals

We always assume zero attacking points for goalkeepers (and don't perform the calculation above).

**Defending Points (Goals Conceded):**

The following is done in `get_defending_points()` in `airsenal.framework.prediction_utils`:
- Calculate clean sheet points for each player (assuming player expected to play 60 mins):
    - For goalkeepers & defenders: 4pts x probability their team concedes zero goals.
    - For midfielders: 1pt x probability their team concedes zero goals.
- Calculate points lost due to goals conceded for each player:
    - For goalkeepers & defenders: -P(2 or 3 goals conceded) - P(4 or 5 goals conceded) etc.

**Final Prediction:**

See `calc_predicted_points()` in `airsenal.framework.prediction_utils`.

The final points prediction for each player in a fixture is the sum of their predicted appearance points, attacking points and defending points (averaged across the different predictions for the different number of minutes the player might play). The predicted points for a player in a _gameweek_ is the sum of their predicted points in all their team's fixtures in that gameweek - which may be two for double gameweeks (or none in blank gameweeks).

We currently don't have predictions for the points contribution from bonus points, save points, yellow and red cards, own goals, or penalty misses and saves.

### Running Points Predictions

Use `airsenal_run_prediction` (previously `airsenal_run_prediction`) from the command-line, which runs the function `main()` in `airsenal.scripts.fill_predictedscore_table`.

## Creating a Team for the Start of the Season

Run `airsenal_make_squad` from a terminal to create a completely new squad (e.g. for the start of the season). This calls the function `main()` in `airsenal.scripts.team_builder`.

## Transfer & Squad Optimisation

Run `airsenal_run_optimization` (previously `run_airsenal_optimization`) to generate transfer suggestions. This calls the function `main()` in `airsenal.scripts.fill_transfersuggestion_table`.

Starting XI, captain & subs

Chips

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
