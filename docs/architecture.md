# How AIrsenal is put together

This page covers where code lives and why. Related pages:

- [where-to-look.md](where-to-look.md): where to start from a command or a task.
- [how-it-works.md](how-it-works.md): the database schema and how points
  predictions are calculated.
- [adding-a-model.md](adding-a-model.md): how to add a model or an optimizer.
- [xg-models.md](xg-models.md): how the defaults in the two xG models were chosen.

## The dependency chain

The package is split into thirteen subpackages of `src/airsenal`, arranged in a
chain. Each may import from the packages listed below it, and never from those
above it.

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

`game/` is at the bottom. It holds the rules of FPL itself, such as what a
position is, what a goal is worth and what `"2122"` means, and it imports
nothing: no other airsenal package, no third-party library, not even a logger.

The layering is enforced by four `import-linter` contracts in `pyproject.toml`,
checked by `uv run lint-imports`:

| contract | what it stops |
|---|---|
| Stage packages form a one-way dependency chain | an import going back up the list. The contract is `exhaustive`, so a new package fails the check until it is added to the chain |
| The database layer does not reach for the FPL API | a query quietly becoming a live API call |
| The database layer neither renders nor talks to the network | `db` reaching for Rich, matplotlib or an HTTP client |
| Only the remote package talks to the network | any other package importing `curl_cffi`, `requests` or `bs4` |

Two rules that import-linter can't express are checked by tests instead:

- `tests/game/test_game_is_plain_python.py` parses each `game/` module and
  allows only standard library imports and imports from elsewhere in `game/`.
- `tests/test_import_side_effects.py` imports every module with network sockets
  and SQLite blocked.

An import in the wrong direction still works at run time, so only
`lint-imports` will catch it. Run it after moving code between packages (the
pre-commit hook also runs it).

## Where new code goes

Ask these questions in order:

1. **Is it part of the rules of Fantasy Premier League?** For example, what a
   goal is worth, what a position is, how season strings work, or what other
   data sources call a club. That goes in `game/`. If it needs a logger or a
   dataframe, it doesn't belong in `game/`.
2. **Is it generic Python code with no airsenal imports?** That goes in
   `core/`.
3. **Otherwise**, put it in the package from the list above whose job it is. If
   it seems to belong to two packages, put it in the lower one, or split it into
   two functions.

Prefer adding a module to an existing package over creating a new subdirectory
that contains a single file.

[CodingConventions.md](../CodingConventions.md) is the definitive version of
these rules, along with the naming and argument-order conventions.

## The map

Every module, in chain order from the bottom up. To start from a task or a
command instead, see [where-to-look.md](where-to-look.md).
`tests/test_navigation_docs.py` fails if a module is missing from this table or
a row names a module that doesn't exist.

| file | what is in it |
|---|---|
| `game/enums.py` | `Position` and `Chip` |
| `game/scoring.py` | FPL's own rules: points per event, `SQUAD_SIZE`, `POINTS_HIT_COST`, free transfer accrual |
| `game/season.py` | which season it is, and how a season is written |
| `game/mappings.py` | what other data sources call clubs, positions and chips |
| `core/caching.py` | the query caches, and `clear_query_caches()` |
| `core/concurrency.py` | starting the transfer search's worker processes (by fork), and the watchdog that reports a stalled worker |
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
| `remote/errors.py` | `RemoteError` and its subclasses, so callers don't depend on the HTTP library's exceptions |
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
| `prediction/team_models/` | one module per team model, plus shared fitting code, and `scorelines.py`, which turns a predicted mean number of goals into probabilities for each goal count |
| `prediction/point_components/` | one module per part of a score (appearance, attacking, bonus, …), plus `PointsConfig` and the breakdown of actual scores used to evaluate them |
| `prediction/points_models/` | `POINTS_MODELS` and `build_points_model`, which pick a points model by name, and `ComponentPointsModel`, the default |
| `prediction/evaluation.py` | scoring a model, or a whole run's points, against what happened |
| `prediction/run.py` | filling the prediction table, and the tag that groups a run's rows |
| `squad/squad.py` | `Squad`: fifteen players and the rules they obey |
| `squad/player.py` | `CandidatePlayer` and `DummyPlayer`: the player objects a `Squad` holds |
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
| `optimization/run_transfers.py` | running a transfer search: everything around the algorithm |
| `optimization/run_squad.py` | running a from-scratch squad build |
| `optimization/persist.py` | writing a plan to the `transfer_suggestion` and `transaction` tables |
| `optimization/transfer_optimizers/` | one module per whole-window search |
| `optimization/strategies/` | one module per way of choosing a gameweek's transfers |
| `optimization/squad_optimizers/` | one module per whole-squad builder |
| `apply/transfers.py` | posting the suggested transfers, and `build_transfer_payload` for checking them without posting |
| `apply/lineup.py` | posting the starting eleven, captain and bench order |
| `pipeline/run.py` | `AIrsenalPipeline`: the swappable models and optimizers, and the stages that use them |
| `pipeline/settings.py` | `PipelineSettings`: a run's options (season, gameweeks, chips, whether to apply transfers, …), as opposed to the models and optimizers it uses |
| `pipeline/replay.py` | replaying a past season, and the `ReplayResult` it scores |
| `cli/` | one module per command or command group; the table in [where-to-look.md](where-to-look.md) says which |
| `cli/options.py` | the option aliases shared across commands |
| `cli/_components.py` | the settings several commands build from the same flags |

Outside the package, `tests/` mostly mirrors the package layout, `tools/` holds
one-off development scripts (installed with the `tools` extra, and
type-checked), and `notebooks/` holds exploratory Jupyter notebooks.

## Constraints on future changes

**Prediction is single-threaded on purpose.** Don't add threading or
multiprocessing to `prediction/run.py`, or to any code that calls a jax-based
model: jax deadlocks under multi-threading. Prediction is fast enough without it.

**The transfer search must use fork.** `core/concurrency.py` forces the `fork`
start method on POSIX systems. The search passes its workers local progress
callbacks, which pickle can't serialise, so with `spawn` (the default on macOS)
the search fails rather than just running slower. Forking must also happen
before jax has been initialised.
