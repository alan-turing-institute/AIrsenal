# Where to look

Where to start when you want to change, add or debug something. The file map in
[architecture.md](architecture.md) works the other way round, from a file to
what's in it. Paths below are relative to `src/airsenal/`.

## From a command

Every command, where it is defined, and the function that does the work.
`tests/test_navigation_docs.py` fails if a command is missing from this table.

| command | defined in | calls |
|---|---|---|
| `airsenal run` | `cli/run.py` | `AIrsenalPipeline.run` in `pipeline/run.py`, which refreshes the database, then calls the `predict` and `optimize` stages below, then optionally `apply` |
| `airsenal db create` | `cli/db.py` | `create_database` in `ingest/init_db.py`, which calls each `ingest/<table>.py` in turn |
| `airsenal db update` | `cli/db.py` | `update_database` in `ingest/update.py` |
| `airsenal db check` | `cli/db.py` | `run_all_checks` in `ingest/checks.py` |
| `airsenal predict` | `cli/predict.py` | `AIrsenalPipeline.predict` → `make_predictedscore_table` in `prediction/run.py`, then `get_top_predicted_points` in `reporting/top_players.py` |
| `airsenal optimize transfers` | `cli/optimize.py` | `AIrsenalPipeline.optimize` → `run_optimization` in `optimization/run_transfers.py` → the transfer optimizer in `optimization/transfer_optimizers/` |
| `airsenal optimize squad` | `cli/optimize.py` | `AIrsenalPipeline.optimize` → `build_new_squad` in `optimization/run_squad.py` → the squad optimizer in `optimization/squad_optimizers/` |
| `airsenal apply transfers` | `cli/apply.py` | `make_transfers` in `apply/transfers.py` (and `set_lineup` after it) |
| `airsenal apply lineup` | `cli/apply.py` | `set_lineup` in `apply/lineup.py` |
| `airsenal replay` | `cli/replay.py` | `run_replays` → `replay_season` in `pipeline/replay.py` |
| `airsenal env get` | `cli/env.py` | `core/env.py` |
| `airsenal env set` | `cli/env.py` | `save_env` in `core/env.py` |
| `airsenal env delete` | `cli/env.py` | `delete_env` in `core/env.py` |
| `airsenal env names` | `cli/env.py` | `AIRSENAL_ENV_KEYS` in `core/env.py` |
| `airsenal dump api` | `cli/dump.py` | `dump_api` in `export/api_dump.py` |
| `airsenal dump db` | `cli/dump.py` | `dump_db` in `export/db_dump.py` |
| `airsenal dump attributes` | `cli/dump.py` | `save_attributes` in `export/attributes.py` |
| `airsenal dump transfermarkt` | `cli/dump.py` | `scrape_transfermarkt` in `remote/transfermarkt.py` |
| `airsenal plot` | `cli/plot.py` | `plot_standings` in `reporting/plots.py` |

Options shared by several commands are defined once in `cli/options.py`. The
flags that build transfer constraints, chip gameweeks and squad scoring are
turned into settings in `cli/_components.py`.

Never run `airsenal apply` while testing: it makes real changes to the FPL
team. Pass `--dry-run`, or test what `build_transfer_payload` returns.

## From a log line

Run a command with `airsenal -v ...` and every log line is prefixed with the
module and line number that wrote it, e.g. `airsenal.prediction.run:112`.

## From a task

### The rules of the game

| I want to… | start at |
|---|---|
| change what a goal, assist, clean sheet or save is worth | `game/scoring.py` |
| change how free transfers accrue, or what a hit costs | `free_transfers_after` and `POINTS_HIT_COST` in `game/scoring.py`. The search applies them in `calc_free_transfers` in `optimization/moves.py`; `get_free_transfers` in `squad/state.py` works out how many the user currently has |
| change squad rules: budget, players per club, players per position | the `check_*` methods of `Squad`, and `TOTAL_PER_POSITION`, in `squad/squad.py`; the budget is in `SquadScoringConfig` in `optimization/squad_score.py` |
| change how chips work | `Chip` in `game/enums.py`, `game/chips.py` for how many a season gives and when they expire, `optimization/moves.py` for when they can be played, `optimization/chip_timing.py` for the rules behind `--chip-heuristic`, and `chip_gameweeks` in `cli/_components.py` for the flags |
| change how the current season is decided | `get_current_season` in `game/season.py` |
| match a club or position name from another data source | `game/mappings.py` |

### The database and the data in it

| I want to… | start at |
|---|---|
| add a column or a table | `db/models.py`, then the `ingest/<table>.py` that fills it. There are no migrations: rebuild with `airsenal db create --clean` |
| add or fix a query | `db/queries/<subject>.py` |
| find out which gameweek AIrsenal thinks it is | `next_gameweek` in `db/queries/gameweeks.py` |
| use a new field from the FPL API | a method on `FPLDataFetcher` in `remote/fpl_api.py`, then the `ingest/` module that stores it |
| debug `db update` failing or missing data | `update_database` in `ingest/update.py`; `airsenal db check` runs the consistency checks in `ingest/checks.py` |
| fix a player whose name is not matched | `get_player` and `get_player_by_similar_name` in `db/queries/players.py`. Genuine spelling variants go in `data/alternative_player_names.csv`, read by `ingest/player_mappings.py` |
| fix one person split into two player rows | `fill_player_table_from_file` in `ingest/players.py` joins a past season's player to an existing row by opta code (`find_player_in_table`), and records that season's name as an alternative. Seasons before 2425 get their opta code from `opta_code` in `export/player_summary.py` |
| change how injuries and suspensions are read | `is_injured_or_suspended` on `Player` in `db/models.py`, filled by `fill_availability_for_season` in `ingest/player_attributes.py` |
| add a new season's packaged data | `src/airsenal/data/`, one file per kind per season (`teams_2627.csv`); `core/data_files.py` finds them, `export/` writes most of them |
| point AIrsenal at a different database | `db/engine.py`, configured through `core/env.py` |

### Predictions

| I want to… | start at |
|---|---|
| add a model or a point component | [adding-a-model.md](adding-a-model.md). It needs one class and one line in a table |
| change the xG team or player model | [xg-models.md](xg-models.md) first: most of the obvious improvements have already been tried and rejected |
| work out why one player's prediction looks wrong | `score_prediction_breakdown` in `prediction/evaluation.py` splits a prediction into its components (defined in `prediction/point_components/`). `get_player_history_df` in `prediction/features.py` returns the data the models were fitted to |
| change how minutes are predicted | `prediction/minutes_models/`, and the helpers in `prediction/minutes.py` |
| find out whether a change made predictions better | the `score_*` and `backtest_*` functions in `prediction/evaluation.py`, or `airsenal replay` for a whole season |
| change what is written to the prediction table, or how runs are tagged | `prediction/run.py`, and `db/queries/tags.py` |

### Transfers and squads

| I want to… | start at |
|---|---|
| change how transfers are chosen in a gameweek | `optimization/strategies/`, one module per strategy |
| change how transfers across several gameweeks are searched | `optimization/transfer_optimizers/`, one module per search; the moves allowed in each gameweek, and what each one scores, are in `branches.py` |
| change how a squad's score over several gameweeks is calculated, e.g. substitute weights or discounting later gameweeks | `optimization/squad_score.py` |
| change the limit on total hits, or other transfer constraints | `optimization/protocols.py`, and `transfer_constraints` in `cli/_components.py` |
| change how a squad is built from scratch | `optimization/squad_optimizers/` |
| change how the starting eleven or captain is picked | `squad/lineup.py` |
| change selling prices | `squad/pricing.py` |
| change what a search saves to the database | `optimization/persist.py` |
| debug a transfer search that hangs or whose workers crash | `core/concurrency.py`, which forks the worker processes and runs a watchdog that reports stalled workers. Read its docstring before changing how workers are started |

### Output, settings and everything else

| I want to… | start at |
|---|---|
| change a table or panel printed in the terminal | `reporting/`: `top_players.py` for predictions, `optimization.py` for searches, `squad_view.py` for squads |
| change a Discord post | `predicted_points_discord_payload` in `reporting/top_players.py` and `discord_payload` in `reporting/optimization.py`, sent by `remote/discord.py` |
| add a command-line option | the command's module in `cli/`. Put it in `cli/options.py` if another command already has it, and in `PipelineSettings` in `pipeline/settings.py` if it changes what a run does |
| add a setting such as a credential | `AIRSENAL_ENV_KEYS` in `core/env.py` |
| debug FPL login | `remote/fpl_auth.py` |
| see stale results after changing the database | `clear_query_caches` in `core/caching.py` |

## When a guardrail test fails

| test | what it checks |
|---|---|
| `tests/test_argument_order.py` | function arguments are in the documented order: see [CodingConventions.md](../CodingConventions.md) |
| `tests/test_naming_conventions.py` | `gameweek` is written out in full, and the `Position` and `Chip` enums are used rather than strings |
| `tests/test_component_tables.py` | every entry in a model or optimizer table can be built with no arguments and has the methods its protocol requires |
| `tests/test_navigation_docs.py` | this page lists every command, and the map in architecture.md lists every module |
| `tests/game/test_game_is_plain_python.py` | `game/` imports nothing outside the standard library and `game/` itself |
| `tests/test_import_side_effects.py` | importing a module touches neither the network nor the database |
| `uv run lint-imports` | no package imports from one above it in the chain in [architecture.md](architecture.md) |

The rest of `tests/` mirrors the package layout, so the tests for
`squad/squad.py` are in `tests/squad/`. `tests/e2e/` runs real model fits and
transfer searches against a small test database.
