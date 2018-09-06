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
