# How AIrsenal works

Where the code lives is [architecture.md](architecture.md); how to add your own model
is [adding-a-model.md](adding-a-model.md). This is what the code does at runtime.

Reference for the parts of AIrsenal that aren't obvious from the code alone: what's in
the database, and how a points prediction is put together. For installation and the
commands to run, see the [README](../README.md).

## Database

The database holds data from the previous three FPL seasons (packaged in the repo at
`src/airsenal/data`) and the current season (from the FPL API), plus the user's own squad
and AIrsenal's predictions and suggestions. Much of the historic data was compiled with
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
  availability (`news`, `chance_of_playing_next_round`, `return_gameweek`) as it stood
  in that gameweek. The availability columns are what every "is this player going to
  play?" question reads, for a past season as well as this one.
  Managers are in here too, as position `MNG`, and nothing in AIrsenal models one:
  `Position.is_modelled` is what every place that reads a position asks, and they are
  left out of the data models are fitted to and skipped rather than predicted.
- **PlayerScore** — per-match stats: points, goals, goals conceded, assists, bonus,
  minutes and others, including the expected goals and assists the FPL API has recorded
  since 2223. A team's expected goals are the sum of its players', which is where both
  xG models get them.

**Squad and AIrsenal data**

- **Transaction** — every player bought and sold in your FPL team, with price and
  gameweek. This is what determines your starting squad and budget for an optimisation.
  Two flags say how a row came about: `free_hit`, meaning the change lasted one
  gameweek and is skipped when the squad is rebuilt, and `counts_as_transfer`, which is
  0 for the fifteen you started with and for anything bought on a wildcard or free hit.
  `get_free_transfers` reads the second to work out what each gameweek starts with. The
  two cases it marks differ afterwards: the count starts at one the gameweek after the
  opening fifteen, whereas a wildcard or free hit freezes it at whatever it had reached
  — you keep the free transfers you had, and gain none for that gameweek.
- **PlayerPrediction** — predicted points per player per fixture from a prediction run.
- **TransferSuggestion** — recommended transfers from an optimisation run.

### Interacting with the FPL API

`FPLDataFetcher` in `airsenal.remote.fpl_api` is the only way AIrsenal reads the FPL API.
It's used for database setup and updates, and elsewhere — for example to check a player's
current injury status during prediction.

### Data sanity checks

`airsenal db check` runs `run_all_checks` in `airsenal.ingest.checks`, which verifies:

- Every season has 20 teams, 3 of them newly promoted, and 380 fixtures.
- Players are assigned to one of the two teams in each fixture they appear in.
- 11 to 14 players appear for each team in each fixture. This fails for the end of the
  1920 season, when 5 substitutes were allowed.
- Player goals and own goals sum to the final score for each team.
- Assists are no more than goals.
- Goals conceded match goals scored by the opponent.

## Player points predictions

Predictions come from three components:

1. A team-level model predicting final score probabilities for each match.
2. A player-level model predicting the probability a player scores or assists each goal
   his team scores.
3. Recent minutes played, and current injury or suspension status.

For background on the modelling, see
[the AIrsenal write-up](https://www.turing.ac.uk/news/airsenal).

The default team model, `xg`, rates each team's attack and defence from the expected
goals in past matches and reads the result as a Conway-Maxwell-Poisson over goal counts;
`extended` and `neutral` are Dixon-Coles models from the
[bpl](https://github.com/anguswilliams91/bpl-next) package, fitted to goals, and are what
to ask for on a season before 2223, when the FPL API began recording expected goals.
The player models in `airsenal.prediction.player_models` divide those goals up, and come
in the same two kinds: `xg`, the default, is a Dirichlet fitted to who was *expected* to
score and assist, and `conjugate` is the same update fitted to who actually did - which
is what a season before 2223 needs, for the same reason `extended` is. All of them are
one module per model, behind the `TeamModel` and `PlayerModel` protocols.

### How predicted points are calculated

Before anything per-player:

- Fit the team and player models.
- Predict the probability of each number of goals scored and conceded by each team in
  each fixture in the window.
- Get each player's minutes in their last three fixtures, and their injury and suspension
  status.

**Recent minutes and appearance points**

`get_recent_minutes_for_player` in `airsenal.prediction.minutes` returns the minutes a
player played in their last 3 matches (by default). Points are predicted once per
distinct value and then averaged — so a player who played 0, 70 and 90 minutes gets three
predictions. Throughout, the probability of scoring, assisting or conceding is weighted by
the fraction of the match the player is assumed to play.

A player marked as having a 50% or lower chance of playing
(`Player.is_injured_or_suspended()` in `airsenal.db.models`) is predicted 0 points. That
is the only availability check, for every season: `PlayerAttributes` carries the flags
throughout. For the gameweek being predicted they come from the FPL API; for a gameweek
already played they come from the per-day attributes history
(`player_attributes_history_yyyy.csv`, read at the date of the gameweek's first match),
falling back to the Transfermarkt-scraped `absences_yyyy.csv` for the seasons and
gameweeks the history does not reach.

The question takes two gameweeks to ask: the gameweek being predicted *from*, and the
gameweek of the fixture. A player only scores 0 if they were already out by the first and
not due back by the second — an absence that began later has not happened yet, and a
replay must not act on it. The flag living on the row for the gameweek it was known in is
what makes that hold. A row with a low chance and no return gameweek means out
indefinitely, which is what the API means by it.

Appearance points follow FPL's rule: 0 for not playing, 1 for under 60 minutes, 2 for 60
or more (`get_appearance_points` in `airsenal.game.scoring`).

**Attacking points**, in `get_attacking_points()`:

- The probability the team scores each number of goals.
- The possible splits of those goals into (player scores, player assists, neither).
- The probability of each split, from the trinomial player model.
- The FPL points each split is worth, given the points for a goal and an assist in the
  player's position — including goalkeepers, who are worth 10 points a goal.
- Multiply the probabilities by the points and sum.

**Defending points**, in `get_defending_points()`:

- Clean sheet points, only for players expected to play 60 minutes or more: 4 points ×
  P(team concedes zero) for goalkeepers and defenders, 1 point for midfielders.
- Points lost to goals conceded, for goalkeepers and defenders only: 1 point per 2 goals
  conceded, scaled by the fraction of the match the player is expected to play.
- Forwards score no defending points.

**The other components**, each fitted as a small empirical model from past seasons and
each switchable with a flag:

| Component | Function | Flag |
|---|---|---|
| Bonus points | `get_bonus_points` | `--no-bonus` |
| Yellow and red cards | `get_card_points` | `--no-cards` |
| Goalkeeper saves | `get_save_points` | `--no-saves` |
| Defensive contributions | `get_def_con_points` | `--no-def-con` |

Turning one off skips fitting it and leaves it out of the total. There are still no
predictions for own goals or penalty misses and saves.

**The final prediction**, in `calc_predicted_points_for_player()`, is the sum of all the
components above, averaged over the different numbers of minutes the player might play.
A player's predicted points for a *gameweek* is the sum over all their team's fixtures in
it — two in a double gameweek, none in a blank.
