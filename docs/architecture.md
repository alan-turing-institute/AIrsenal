# How AIrsenal is put together

Where things live, and why they live there. To start from a command or a task
instead, see [where-to-look.md](where-to-look.md). For what the code *does* at
runtime - the database schema and how points predictions are built - see
[how-it-works.md](how-it-works.md). For how to add a model or an algorithm, see
[adding-a-model.md](adding-a-model.md), and for where every number in the two
default models came from, [xg-models.md](xg-models.md).

## The dependency chain

The package is thirteen subdirectories of `src/airsenal`, forming a one-way
chain. Each may import from the rows below it and never from the rows above.

```
cli            command definitions and argument parsing, nothing else
pipeline       orchestration: `run` and `replay`
apply          the only code that writes to the real FPL entry
optimization   the transfer search and the whole-squad builder
export         writing data back out
ingest         filling the database from packaged data and the FPL API
reporting      rendering results: tables, plots, Discord posts
squad          the Squad class and the state of the user's own entry
prediction     the models, the points they imply, and how to score a model
db             the tables, the queries and the session
remote         everything that talks to the internet, and nothing else
core           generic plumbing: logging, caching, the console, date parsing
game           the facts about FPL - it imports nothing at all
```

`game/` is last, and that is the point: what a position is, what a goal is worth,
what `"2122"` means. It imports nothing - not another airsenal package, not a
third-party library, not even a logger.

Four `import-linter` contracts in `pyproject.toml` enforce this, and
`uv run lint-imports` checks them:

| contract | what it stops |
|---|---|
| Stage packages form a one-way dependency chain | an import going back up the list. `exhaustive`, so a new package must be given a place rather than escaping the chain |
| The database layer does not reach for the FPL API | a query quietly becoming a live API call |
| The database layer neither renders nor talks to the network | `db` reaching for Rich, matplotlib or an HTTP client |
| Only the remote package talks to the network | any other package importing `curl_cffi`, `requests` or `bs4` |

Two things the contracts cannot see are checked by tests instead:
`tests/game/test_game_is_plain_python.py` walks each `game/` module's syntax tree
and allows only stdlib and sibling imports, and `tests/test_import_side_effects.py`
imports every module with sockets and SQLite blocked.

A wrong-direction import does not fail at run time, so run `uv run lint-imports`
yourself after moving code between packages. The pre-commit hook also runs it.

## Where new code goes

Three questions, in order:

1. **Is it a fact about Fantasy Premier League?** What a goal is worth, what a
   position is, how season strings work, what other data sources call a club -
   that goes in `game/`. Needing a logger or a dataframe is the signal that it is
   not a fact about the game.
2. **Is it generic Python machinery with no airsenal imports?** That goes in
   `core/`.
3. **Otherwise it belongs to the stage that owns it** - the list above. If it
   seems to belong to two stages, it goes in the lower one, or it is two
   functions.

Prefer a module in an existing package to a new subdirectory holding one file: a
directory tells the reader a category exists without telling them what is in it.

`CodingConventions.md` at the repository root is the canonical version of this
rule, along with the naming and argument-order conventions.

## The map

Every module, in the order of the chain from the bottom up. To start from a task
or a command instead, see [where-to-look.md](where-to-look.md).
`tests/test_navigation_docs.py` fails when a module is missing from this table
or a row names one that does not exist.

| file | what is in it |
|---|---|
| `game/enums.py` | `Position` and `Chip` |
| `game/scoring.py` | FPL's own rules: points per event, `SQUAD_SIZE`, `POINTS_HIT_COST`, free transfer accrual |
| `game/season.py` | which season it is, and how a season is written |
| `game/chips.py` | how many of each chip a season gives, and which are used up by a gameweek |
| `game/mappings.py` | what other data sources call clubs, positions and chips |
| `core/caching.py` | the query caches, and `clear_query_caches()` |
| `core/concurrency.py` | the fork the transfer search needs, and the handlers that make it safe |
| `core/console.py` | Rich console, tables, progress bars, and `confirm()` |
| `core/copy.py` | fast deep copies for the optimiser's inner loop |
| `core/data_files.py` | locating the packaged data in `src/airsenal/data/` |
| `core/dates.py` | date and datetime parsing |
| `core/env.py` | AIrsenal's settings (`FPL_TEAM_ID`, `FPL_LOGIN`, the database) and where they are stored |
| `core/logging.py` | logger setup, and relaying a forked worker's logs through its parent |
| `core/lookup.py` | `lookup()` and `ConfigError`: turning a name into an implementation |
| `remote/fpl_api.py` | `FPLDataFetcher`, the FPL API client, and `get_fetcher()` |
| `remote/fpl_http.py` | where the FPL API lives and how a request to it is made |
| `remote/fpl_auth.py` | logging in to the FPL account |
| `remote/errors.py` | `RemoteError` and friends, so callers need not know the HTTP library |
| `remote/transfermarkt.py` | scraping injuries and suspensions from Transfermarkt |
| `remote/discord.py` | posting to a Discord webhook |
| `remote/download.py` | resumable file downloads |
| `db/models.py` | every table in the database |
| `db/queries/` | reading and writing them, one module per subject |
| `db/session.py` | the lazily-created engine and the default session |
| `db/engine.py` | which database to talk to: SQLite, or postgres from `AIRSENAL_DB_URI` |
| `prediction/protocols.py` | `PointsModel`, `PlayerModel`, the three `TeamModel` kinds, `MinutesModel`, `PointComponent`, and the typed data and requests each is given |
| `prediction/features.py` | assembling the historical data the models are fitted to |
| `prediction/minutes.py` | recent minutes, and whether a player is absent, for the minutes models |
| `prediction/player_models/` | one module per player model, plus shared fitting and scaling |
| `prediction/minutes_models/` | one module per way of predicting how long a player plays |
| `prediction/team_models/` | one module per team model, plus shared fitting, and `scorelines.py` for adapting between what a model predicts and what prediction needs |
| `prediction/point_components/` | one module per part of a score, plus `PointsConfig` and the realised-score breakdown that scores them |
| `prediction/points_models/` | the seam the rest of prediction sits behind, and `ComponentPointsModel` |
| `prediction/evaluation.py` | scoring a model, or a whole run's points, against what happened |
| `prediction/run.py` | filling the prediction table, and the tag that groups a run's rows |
| `squad/squad.py` | `Squad`: fifteen players and the rules they obey |
| `squad/player.py` | `CandidatePlayer` and `DummyPlayer`: a player as the squad sees them |
| `squad/lineup.py` | choosing the starting eleven, bench order and captain |
| `squad/pricing.py` | what a player would sell for |
| `squad/state.py` | the state of the user's own entry: bank, free transfers, players and chips by gameweek |
| `squad/history.py` | the user's transaction history, rebuilt from the FPL API |
| `reporting/top_players.py` | the top predicted players, as tables and Discord posts |
| `reporting/optimization.py` | printing what a search decided, and its Discord post |
| `reporting/squad_view.py` | drawing a squad in the terminal |
| `reporting/plots.py` | a mini-league's standings by gameweek |
| `ingest/init_db.py` | `airsenal db create`: building the database from scratch |
| `ingest/update.py` | `airsenal db update`: bringing it up to date with the FPL API |
| `ingest/checks.py` | `airsenal db check`: consistency checks over the result |
| `ingest/teams.py` | filling `team` |
| `ingest/players.py` | filling `player` |
| `ingest/player_mappings.py` | filling `player_mapping`, the other names a player goes by |
| `ingest/player_attributes.py` | filling `player_attributes`: price, team, position and availability per gameweek |
| `ingest/attributes_history.py` | reading the daily player attributes snapshots |
| `ingest/absences.py` | reading the packaged absences CSVs |
| `ingest/fixtures.py` | filling `fixture` |
| `ingest/results.py` | filling `result` |
| `ingest/player_scores.py` | filling `player_score` |
| `ingest/fifa_ratings.py` | filling `fifa_rating` |
| `export/db_dump.py` | `airsenal dump db`: every table to CSV |
| `export/api_dump.py` | `airsenal dump api`: saving what the API returns to the packaged data |
| `export/attributes.py` | `airsenal dump attributes`: today's attributes appended to the season history |
| `export/player_details.py` | every player's gameweek scores this season, for the packaged data |
| `export/player_summary.py` | the player summary files, for the packaged data |
| `export/results.py` | the results CSVs, for the packaged data |
| `optimization/protocols.py` | `SquadOptimizer`, `TransferStrategy`, `TransferOptimizer` and their requests |
| `optimization/moves.py` | what can be done in one gameweek: transfers, chips, hits and free transfers |
| `optimization/plan.py` | `Plan` and `TransferSearchResult`: what a search produces |
| `optimization/squad_score.py` | what a squad is worth over a window, and `SquadScoringConfig` |
| `optimization/chip_timing.py` | the rules that decide chip gameweeks for `--chip-heuristic` |
| `optimization/run_transfers.py` | running a transfer search: everything around the algorithm |
| `optimization/run_squad.py` | running a from-scratch squad build |
| `optimization/persist.py` | writing a plan to the `transfer_suggestion` and `transaction` tables |
| `optimization/transfer_optimizers/` | one module per whole-window search |
| `optimization/strategies/` | one module per way of choosing a gameweek's transfers |
| `optimization/squad_optimizers/` | one module per whole-squad builder |
| `apply/transfers.py` | posting the suggested transfers, and `build_transfer_payload` for checking them without posting |
| `apply/lineup.py` | posting the starting eleven, captain and bench order |
| `pipeline/run.py` | `AIrsenalPipeline`: the swappable components and the stages that use them |
| `pipeline/settings.py` | `PipelineSettings`: what a run does, as opposed to what it does it with |
| `pipeline/replay.py` | replaying a past season, and the `ReplayResult` it scores |
| `cli/` | one module per command or command group; the table in [where-to-look.md](where-to-look.md) says which |
| `cli/options.py` | the option aliases shared across commands |
| `cli/_components.py` | the settings several commands build from the same flags |

Outside the package: `tests/` mirrors it where there is enough to mirror,
`tools/` holds dev one-offs (installed with the `tools` extra, and type-checked),
and `notebooks/` holds exploratory Jupyter notebooks.

## Two things that constrain future work

**Prediction is single-threaded on purpose.** Do not add threading or
multiprocessing to `prediction/run.py`, or to any code that calls a jax-based
model: jax deadlocks under multi-threading. Prediction is fast enough without it.

**The transfer search must fork.** `core/concurrency.py` forces the `fork` start
method on posix. The search hands its workers local progress callbacks, which
pickle cannot serialise, so under `spawn` - macOS's default - it does not run
slower, it fails. It can also only fork before jax has been initialised.
