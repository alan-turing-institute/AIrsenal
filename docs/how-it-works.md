# How AIrsenal works

What's in the database, and how a points prediction is calculated. For installation
and the commands to run, see the [README](../README.md). For where the code lives,
see [architecture.md](architecture.md), and for how to add your own model, see
[adding-a-model.md](adding-a-model.md).

## Database

The database holds data from the previous three FPL seasons (packaged in the repo at
`src/airsenal/data`) and the current season (from the FPL API), plus the user's own squad
and AIrsenal's predictions and suggestions. Much of the historical data was compiled with
the help of [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League).

The schema is defined with `sqlalchemy` in `airsenal.db.models`.

**Football club data**

- **Team** — short and full name, and FPL ID, for each team in each season.
- **Fixture** — every past and future scheduled fixture: home team, away team, date,
  gameweek, season.
- **Result** — home and away goals for each fixture that has been played.
- **FifaTeamRating** — FIFA (the game) team ratings from https://www.fifaindex.com/teams/,
  used mainly to estimate the strength of promoted teams.

**Player data**

- **Player** — name and ID for every player who has been in the game in the last three
  seasons. For players active in the current season the ID matches the FPL player ID.
- **PlayerAttributes** — per-gameweek attributes: FPL price, position, team, and
  availability (`news`, `chance_of_playing_next_round`, `return_gameweek`) as it was
  in that gameweek. Everything that needs to know whether a player is available reads
  these availability columns, for past seasons as well as the current one.
  Managers are included too, with position `MNG`, but AIrsenal doesn't model them.
  Code that reads a position checks `Position.is_modelled`, so managers are left out of
  the data the models are fitted to and are not given predictions.
- **PlayerScore** — per-match stats: points, goals, goals conceded, assists, bonus,
  minutes and others, including the expected goals and assists the FPL API has recorded
  since 2223. Both xG models get a team's expected goals by summing its players' values.
  `news` and `chance_of_playing` are the player's availability on the morning of the
  match, which is used to leave matches a player missed while unavailable out of their
  recent minutes.

**Squad and AIrsenal data**

- **Transaction** — every player bought and sold in your FPL team, with price and
  gameweek. This determines your starting squad and budget for an optimisation.
  Two flags describe each row:
  - `free_hit`: the change only lasted one gameweek, so it is skipped when the squad
    is rebuilt.
  - `counts_as_transfer`: 0 for the fifteen players you started with, and for anything
    bought on a wildcard or free hit.

  `get_free_transfers` uses `counts_as_transfer` to work out how many free transfers
  each gameweek starts with. The two cases where it is 0 are treated differently. After
  the initial squad is picked, the count starts at one in the following gameweek. A
  wildcard or free hit keeps the count where it was: you keep the free transfers you
  had, but don't gain one for that gameweek.
- **PlayerPrediction** — predicted points per player per fixture from a prediction run.
- **TransferSuggestion** — recommended transfers from an optimisation run.

### Interacting with the FPL API

All reads from the FPL API go through `FPLDataFetcher` in `airsenal.remote.fpl_api`.
It's used to set up and update the database, to read the state of your own team, and by
`airsenal apply`. Prediction doesn't use it: availability comes from the database.

### Data sanity checks

`airsenal db check` runs `run_all_checks` in `airsenal.ingest.checks`, which checks that:

- Every season has 20 teams, 3 of them newly promoted, and 380 fixtures.
- Players are assigned to one of the two teams in each fixture they appear in.
- 11 to 14 players appear for each team in each fixture. This fails for the end of the
  1920 season, when 5 substitutes were allowed.
- Player goals and own goals sum to the final score for each team.
- Assists are no more than goals.
- Goals conceded match goals scored by the opponent.

## Player points predictions

The default points model is `ComponentPointsModel` in
`airsenal.prediction.points_models`. It is built from:

1. A team model, which predicts the probability of each team scoring each possible
   number of goals in each match.
2. A player model, which predicts each player's share of their team's goals: the chance
   they score or assist any one of them.
3. A minutes model, which predicts how long each player is on the pitch.
4. The point components, which turn those predictions into FPL points.

For background on the modelling, see
[the AIrsenal write-up](https://www.turing.ac.uk/news/airsenal).

There are two kinds of team and player model: ones fitted to expected goals (xG), and
ones fitted to actual goals. The xG models are the default, but they can only be used
from 2223 onwards, when the FPL API began recording expected goals. For earlier
seasons, use the goals-based models.

- **Team models.** The default, `xg`, rates each team's attack and defence from the
  expected goals in past matches, and uses a Conway-Maxwell-Poisson distribution to
  turn that into probabilities for each number of goals. `extended` and `neutral` are
  Dixon-Coles models from the [bpl](https://github.com/anguswilliams91/bpl-next)
  package, fitted to actual goals.
- **Player models** (in `airsenal.prediction.player_models`) divide a team's goals
  between its players. The default, `xg`, fits a Dirichlet distribution to which
  players were *expected* to score and assist. `conjugate` is the same model fitted to
  who actually scored and assisted.

Each model is its own module, and implements one of the protocols in
`airsenal.prediction.protocols`.

### How predicted points are calculated

First, the team model, player model and any point components that need fitting are
fitted using the data available at the first gameweek being predicted. The team model
then predicts goal probabilities for every fixture in the gameweeks being predicted.

**Minutes and availability**

The default minutes model is `RecentMinutesModel` in
`airsenal.prediction.minutes_models`. It takes the minutes a player played in their
last few matches (one per gameweek being predicted, and at least three) and treats
each as equally likely. If a player hasn't played that many matches this season, the
gap is filled with their average minutes last season for the team they're at now. Points are predicted for each of those minutes values and averaged, so a
player who played 0, 70 and 90 minutes gets three predictions. In each one, the
probability of scoring, assisting or conceding is scaled by the fraction of the match
the player is assumed to play.

A player with a 50% or lower chance of playing (`Player.is_injured_or_suspended()` in
`airsenal.db.models`, used by `is_absent` in `airsenal.prediction.minutes`) is
predicted to play zero minutes, and so to score 0 points. This is the only availability
check, and it works the same way for every season because `PlayerAttributes` always has
the availability columns:

- For the gameweek being predicted, availability comes from the FPL API.
- For past gameweeks, it comes from the daily attributes history
  (`player_attributes_history_yyyy.csv`), as it was on the day of the gameweek's first
  match. Where the history doesn't cover a season or gameweek, it falls back to the
  absences scraped from Transfermarkt (`absences_yyyy.csv`).

Two gameweeks are involved in the check: the gameweek the prediction is made in, and
the gameweek of the fixture. A player is only predicted 0 points if they were already
unavailable at the first and not expected back by the second. An injury that happened
later hadn't happened yet when the prediction was made, so a replay of a past season
must not use it. Storing availability against the gameweek it was reported in is what
makes this possible. A low chance of playing with no return gameweek means the player
is out indefinitely, which is how the FPL API uses it.

**Appearance points** follow FPL's rule: 0 for not playing, 1 for under 60 minutes, 2
for 60 or more (`get_appearance_points` in `airsenal.game.scoring`).

**Attacking points** are calculated in `get_attacking_points()` in
`airsenal.prediction.point_components.attacking`, by:

1. Taking the probability of the team scoring each number of goals.
2. Listing the ways those goals could be split into ones the player scores, ones they
   assist, and ones they are not involved in.
3. Calculating the probability of each split from the player's shares.
4. Working out the FPL points each split is worth, using the points for a goal and an
   assist in the player's position (goalkeepers get 10 points for a goal).
5. Multiplying the probabilities by the points and summing.

**Defending points**, in `get_defending_points()` in
`airsenal.prediction.point_components.defending`:

- Clean sheet points, only for players expected to play 60 minutes or more: 4 points ×
  P(team concedes zero) for goalkeepers and defenders, 1 point for midfielders.
- Points lost for goals conceded, for goalkeepers and defenders only: 1 point per 2
  goals conceded, scaled by the fraction of the match the player is expected to play.
- Forwards score no defending points.

**The other components** each predict the player's past average for that part of the
game, shrunk towards the average for their position. Each can be switched off with a
flag:

| Component | Module in `prediction/point_components/` | Flag |
|---|---|---|
| Bonus points | `bonus.py` | `--no-bonus` |
| Yellow and red cards | `cards.py` | `--no-cards` |
| Goalkeeper saves | `saves.py` | `--no-saves` |
| Defensive contributions | `def_con.py` | `--no-def-con` |

A component that is switched off isn't fitted and is left out of the total. Own goals,
penalties saved and penalties missed are not predicted.

**The final prediction**, in `ComponentPointsModel.predict()`, is the sum of all the
components above, each averaged over the minutes values the player might play.
A player's predicted points for a *gameweek* is the sum over all their team's fixtures in
it: two in a double gameweek, none in a blank.
