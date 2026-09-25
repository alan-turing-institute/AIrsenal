# Where did it go? `main` → `feature/refactor`

Every public function, class and method on `main` (under `airsenal/framework/` and `airsenal/scripts/`), and where it lives now. New paths are relative to `src/airsenal/` unless they start with `tests/` or `tools/`.

331 of 377 names moved or were renamed; 46 were removed. Names that kept their name were matched mechanically; the rest were traced through the commit history on this branch.

## Commands

| old command | new command |
|---|---|
| `airsenal_run_pipeline` | `airsenal run` |
| `airsenal_setup_initial_db` | `airsenal db create` |
| `airsenal_update_db` | `airsenal db update` |
| `airsenal_run_prediction` | `airsenal predict` |
| `airsenal_make_squad` | `airsenal optimize squad` |
| `airsenal_run_optimization` | `airsenal optimize transfers` |
| `airsenal_make_transfers` | `airsenal apply transfers` |
| `airsenal_set_lineup` | `airsenal apply lineup` |
| `airsenal_replay_season` | `airsenal replay` |
| `airsenal_env` | `airsenal env get / set / delete / names` |
| `airsenal_check_data` | `airsenal db check` |
| `airsenal_dump_api` | `airsenal dump api` |
| `airsenal_dump_db` | `airsenal dump db` |
| `airsenal_scrape_transfermarkt` | `airsenal dump transfermarkt` |
| `airsenal_plot` | `airsenal plot` |
| `airsenal_save_absences` | removed: the Absence table is gone; availability is on PlayerAttributes |

## Modules

Where each old module's contents went, most first.

| old module | now in |
|---|---|
| `airsenal/conftest.py` | `tests/conftest.py` (3), `db/session.py` (1) |
| `airsenal/framework/FPL_scoring_rules.py` | `game/scoring.py` (1) |
| `airsenal/framework/aws_utils.py` | removed (5) |
| `airsenal/framework/bpl_interface.py` | `prediction/team_models/fitting.py` (7), removed (1), `prediction/team_models/__init__.py` (1) |
| `airsenal/framework/data_fetcher.py` | `remote/fpl_api.py` (22), `remote/fpl_auth.py` (3), removed (1) |
| `airsenal/framework/env.py` | `core/env.py` (4) |
| `airsenal/framework/fpl_team_utils.py` | removed (3) |
| `airsenal/framework/multiprocessing_utils.py` | `core/concurrency.py` (9) |
| `airsenal/framework/optimization_squad.py` | `optimization/squad_optimizers/genetic_algorithm.py` (3) |
| `airsenal/framework/optimization_transfers.py` | `optimization/strategies/single.py` (1), `optimization/strategies/double.py` (1), `optimization/strategies/random_search.py` (1), `optimization/transfer_optimizers/branches.py` (1) |
| `airsenal/framework/optimization_utils.py` | `optimization/persist.py` (4), `optimization/moves.py` (2), `squad/history.py` (2), `optimization/squad_score.py` (2), removed (2), `optimization/transfer_optimizers/branches.py` (2), `db/queries/tags.py` (1), `optimization/plan.py` (1), `optimization/protocols.py` (1) |
| `airsenal/framework/player.py` | `squad/player.py` (4), removed (2) |
| `airsenal/framework/player_model.py` | `prediction/player_models/conjugate.py` (5), `prediction/protocols.py` (3), removed (3), `prediction/player_models/mcmc.py` (3), `prediction/player_models/scaling.py` (2) |
| `airsenal/framework/prediction_utils.py` | `db/queries/scores.py` (4), removed (3), `prediction/features.py` (2), `prediction/point_components/bonus.py` (2), `prediction/point_components/def_con.py` (2), `prediction/point_components/saves.py` (2), `prediction/point_components/cards.py` (2), `prediction/player_models/fitting.py` (2), `prediction/point_components/attacking.py` (1), `prediction/point_components/defending.py` (1), `prediction/points_models/component.py` (1), `prediction/run.py` (1), `prediction/point_components/empirical_bayes.py` (1) |
| `airsenal/framework/random_team_model.py` | `prediction/team_models/random_model.py` (4) |
| `airsenal/framework/schema.py` | `db/models.py` (17), removed (3), `db/session.py` (3), `remote/transfermarkt.py` (1), `db/engine.py` (1), `db/queries/teams.py` (1) |
| `airsenal/framework/season.py` | `game/season.py` (3), `db/queries/teams.py` (1), `remote/transfermarkt.py` (1) |
| `airsenal/framework/squad.py` | `squad/squad.py` (17), `squad/lineup.py` (6) |
| `airsenal/framework/transaction_utils.py` | `db/queries/transactions.py` (3), `squad/history.py` (2), `squad/state.py` (1) |
| `airsenal/framework/utils.py` | removed (8), `db/queries/gameweeks.py` (7), `db/queries/fixtures.py` (6), `db/queries/scores.py` (6), `db/queries/players.py` (5), `squad/state.py` (4), `game/season.py` (3), `prediction/minutes.py` (3), `core/dates.py` (2), `db/queries/predictions.py` (2), `reporting/top_players.py` (2), `db/queries/tags.py` (2), `remote/transfermarkt.py` (1), `db/queries/teams.py` (1), `export/player_details.py` (1), `ingest/player_attributes.py` (1), `db/models.py` (1), `remote/fpl_api.py` (1), `core/copy.py` (1) |
| `airsenal/scripts/airsenal_run_pipeline.py` | `pipeline/run.py` (6), `optimization/moves.py` (1), `ingest/update.py` (1) |
| `airsenal/scripts/data_sanity_checks.py` | `ingest/checks.py` (10) |
| `airsenal/scripts/dump_db_contents.py` | `export/db_dump.py` (2) |
| `airsenal/scripts/duplicate_names.py` | `tools/duplicate_names.py` (1) |
| `airsenal/scripts/fill_absence_table.py` | `ingest/absences.py` (1), `ingest/player_attributes.py` (1) |
| `airsenal/scripts/fill_db_init.py` | `ingest/init_db.py` (2), removed (1) |
| `airsenal/scripts/fill_fifa_ratings_table.py` | `ingest/fifa_ratings.py` (1) |
| `airsenal/scripts/fill_fixture_table.py` | `ingest/fixtures.py` (3) |
| `airsenal/scripts/fill_player_attributes_table.py` | `ingest/player_attributes.py` (3) |
| `airsenal/scripts/fill_player_mappings_table.py` | `ingest/player_mappings.py` (3) |
| `airsenal/scripts/fill_player_table.py` | `ingest/players.py` (6), removed (1) |
| `airsenal/scripts/fill_playerscore_table.py` | `ingest/player_scores.py` (4), `remote/download.py` (1), `ingest/attributes_history.py` (1) |
| `airsenal/scripts/fill_predictedscore_table.py` | `prediction/run.py` (2) |
| `airsenal/scripts/fill_result_table.py` | `ingest/results.py` (3) |
| `airsenal/scripts/fill_team_table.py` | `ingest/teams.py` (2) |
| `airsenal/scripts/fill_transfersuggestion_table.py` | removed (3), `reporting/optimization.py` (2), `optimization/run_transfers.py` (2), `optimization/transfer_optimizers/tree_search.py` (1), `optimization/moves.py` (1), `cli/optimize.py` (1) |
| `airsenal/scripts/get_transfer_suggestions.py` | `db/queries/predictions.py` (1), removed (1) |
| `airsenal/scripts/make_player_details.py` | `export/player_details.py` (4), `db/queries/fixtures.py` (1) |
| `airsenal/scripts/make_player_history_table.py` | removed (1) |
| `airsenal/scripts/make_player_summary.py` | `export/player_summary.py` (1) |
| `airsenal/scripts/make_results.py` | `export/results.py` (1) |
| `airsenal/scripts/make_transfers.py` | `apply/transfers.py` (12), `apply/lineup.py` (1) |
| `airsenal/scripts/match_player_names.py` | removed (1) |
| `airsenal/scripts/match_team_names.py` | removed (1) |
| `airsenal/scripts/plot_league_standings.py` | `reporting/plots.py` (3) |
| `airsenal/scripts/replay_season.py` | `pipeline/replay.py` (3) |
| `airsenal/scripts/save_attributes.py` | `export/attributes.py` (2), `db/queries/gameweeks.py` (1) |
| `airsenal/scripts/save_expected_absences.py` | removed (2) |
| `airsenal/scripts/scrape_transfermarkt.py` | `remote/transfermarkt.py` (14), `db/queries/teams.py` (1) |
| `airsenal/scripts/scrape_understat.py` | removed (3) |
| `airsenal/scripts/set_env.py` | `cli/env.py` (2) |
| `airsenal/scripts/set_lineup.py` | `apply/lineup.py` (4), `apply/transfers.py` (1), removed (1) |
| `airsenal/scripts/squad_builder.py` | `optimization/run_squad.py` (1) |
| `airsenal/scripts/tune_player_time_weighting.py` | `tools/tune_player_time_weighting.py` (2), `prediction/evaluation.py` (1) |
| `airsenal/scripts/tune_team_time_weighting.py` | `tools/tune_team_time_weighting.py` (2) |
| `airsenal/scripts/update_db.py` | `ingest/update.py` (6) |

## Every name

| old module | old name | now | note |
|---|---|---|---|
| `airsenal/conftest.py` | `session_scope` | `db/session.py::session_scope` | same name |
| `airsenal/conftest.py` | `past_data_session_scope` | `tests/conftest.py::past_data_session_scope` | moved with tests to repo root |
| `airsenal/conftest.py` | `value_generator` | `tests/conftest.py::value_generator` | moved with tests to repo root |
| `airsenal/conftest.py` | `fill_players` | `tests/conftest.py::fill_players` | moved with tests to repo root |
| `airsenal/framework/FPL_scoring_rules.py` | `get_appearance_points` | `game/scoring.py::get_appearance_points` | same name |
| `airsenal/framework/aws_utils.py` | `download_sqlite_file` | removed | removed: Alexa/AWS helpers dropped, no importers |
| `airsenal/framework/aws_utils.py` | `get_league_standings_string` | removed | removed: Alexa/AWS helpers dropped, no importers |
| `airsenal/framework/aws_utils.py` | `get_suggestions_string` | removed | removed: Alexa/AWS helpers dropped, no importers |
| `airsenal/framework/aws_utils.py` | `build_suggestion_string` | removed | removed: Alexa/AWS helpers dropped, no importers |
| `airsenal/framework/aws_utils.py` | `get_score_ranking_string` | removed | removed: Alexa/AWS helpers dropped, no importers |
| `airsenal/framework/bpl_interface.py` | `get_result_dict` | `prediction/team_models/fitting.py::get_result_dict` | same name |
| `airsenal/framework/bpl_interface.py` | `get_ratings_dict` | `prediction/team_models/fitting.py::get_ratings_dict` | same name |
| `airsenal/framework/bpl_interface.py` | `get_training_data` | `prediction/team_models/fitting.py::get_training_data` | same name |
| `airsenal/framework/bpl_interface.py` | `create_and_fit_team_model` | `prediction/team_models/fitting.py::get_fitted_team_model` | inlined as model.fit(); models hold their own fit args |
| `airsenal/framework/bpl_interface.py` | `add_new_teams_to_model` | `prediction/team_models/fitting.py::add_new_teams_to_model` | same name |
| `airsenal/framework/bpl_interface.py` | `get_fitted_team_model` | `prediction/team_models/fitting.py::get_fitted_team_model` | same name |
| `airsenal/framework/bpl_interface.py` | `fixture_probabilities` | removed | removed: callers use TeamModel.predict_outcome_proba directly |
| `airsenal/framework/bpl_interface.py` | `get_goal_probabilities_for_fixtures` | `prediction/team_models/fitting.py::get_goal_probabilities_for_fixtures` | same name |
| `airsenal/framework/bpl_interface.py` | `parse_team_model_from_str` | `prediction/team_models/__init__.py::build_team_model` | replaced by TEAM_MODELS name-to-factory table |
| `airsenal/framework/data_fetcher.py` | `generate_code_verifier` | `remote/fpl_auth.py::generate_code_verifier` | same name |
| `airsenal/framework/data_fetcher.py` | `generate_code_challenge` | `remote/fpl_auth.py::generate_code_challenge` | same name |
| `airsenal/framework/data_fetcher.py` | `FPLDataFetcher` | `remote/fpl_api.py::FPLDataFetcher` | same name |
| `airsenal/framework/data_fetcher.py` | `FPLDataFetcher.get_fpl_credentials` | `remote/fpl_auth.py::FPLAuth.get_fpl_credentials` | login flow split out into FPLAuth |
| `airsenal/framework/data_fetcher.py` | `FPLDataFetcher.login` | `remote/fpl_api.py::FPLDataFetcher.login` | same name |
| `airsenal/framework/data_fetcher.py` | `FPLDataFetcher.get_current_squad_data` | `remote/fpl_api.py::FPLDataFetcher.get_current_squad_data` | same name |
| `airsenal/framework/data_fetcher.py` | `FPLDataFetcher.get_current_picks` | `remote/fpl_api.py::FPLDataFetcher.get_current_picks` | same name |
| `airsenal/framework/data_fetcher.py` | `FPLDataFetcher.get_num_free_transfers` | `remote/fpl_api.py::FPLDataFetcher.get_num_free_transfers` | same name |
| `airsenal/framework/data_fetcher.py` | `FPLDataFetcher.get_current_bank` | `remote/fpl_api.py::FPLDataFetcher.get_current_bank` | same name |
| `airsenal/framework/data_fetcher.py` | `FPLDataFetcher.get_available_chips` | `remote/fpl_api.py::FPLDataFetcher.get_available_chips` | same name |
| `airsenal/framework/data_fetcher.py` | `FPLDataFetcher.get_current_summary_data` | `remote/fpl_api.py::FPLDataFetcher.get_current_summary_data` | same name |
| `airsenal/framework/data_fetcher.py` | `FPLDataFetcher.get_fpl_team_data` | `remote/fpl_api.py::FPLDataFetcher.get_fpl_team_data` | same name |
| `airsenal/framework/data_fetcher.py` | `FPLDataFetcher.get_fpl_team_history_data` | `remote/fpl_api.py::FPLDataFetcher.get_fpl_team_history_data` | same name |
| `airsenal/framework/data_fetcher.py` | `FPLDataFetcher.get_fpl_transfer_data` | `remote/fpl_api.py::FPLDataFetcher.get_fpl_transfer_data` | same name |
| `airsenal/framework/data_fetcher.py` | `FPLDataFetcher.get_fpl_league_data` | `remote/fpl_api.py::FPLDataFetcher.get_fpl_league_data` | same name |
| `airsenal/framework/data_fetcher.py` | `FPLDataFetcher.get_event_data` | `remote/fpl_api.py::FPLDataFetcher.get_event_data` | same name |
| `airsenal/framework/data_fetcher.py` | `FPLDataFetcher.get_player_summary_data` | `remote/fpl_api.py::FPLDataFetcher.get_player_summary_data` | same name |
| `airsenal/framework/data_fetcher.py` | `FPLDataFetcher.get_current_team_data` | `remote/fpl_api.py::FPLDataFetcher.get_current_team_data` | same name |
| `airsenal/framework/data_fetcher.py` | `FPLDataFetcher.get_gameweek_data_for_player` | `remote/fpl_api.py::FPLDataFetcher.get_gameweek_data_for_player`<br>`remote/fpl_api.py::FPLDataFetcher.get_gameweek_data_for_player`<br>`remote/fpl_api.py::FPLDataFetcher.get_gameweek_data_for_player` | same name, defined in several places |
| `airsenal/framework/data_fetcher.py` | `FPLDataFetcher.get_fixture_data` | `remote/fpl_api.py::FPLDataFetcher.get_fixture_data` | same name |
| `airsenal/framework/data_fetcher.py` | `FPLDataFetcher.get_transfer_deadlines` | removed | removed: dead, only caller was is_transfer_deadline_today |
| `airsenal/framework/data_fetcher.py` | `FPLDataFetcher.get_lineup` | `remote/fpl_api.py::FPLDataFetcher.get_lineup` | same name |
| `airsenal/framework/data_fetcher.py` | `FPLDataFetcher.post_lineup` | `remote/fpl_api.py::FPLDataFetcher.post_lineup` | same name |
| `airsenal/framework/data_fetcher.py` | `FPLDataFetcher.post_transfers` | `remote/fpl_api.py::FPLDataFetcher.post_transfers` | same name |
| `airsenal/framework/env.py` | `check_valid_key` | `core/env.py::check_valid_key` | same name |
| `airsenal/framework/env.py` | `save_env` | `core/env.py::save_env` | same name |
| `airsenal/framework/env.py` | `delete_env` | `core/env.py::delete_env` | same name |
| `airsenal/framework/env.py` | `get_env` | `core/env.py::get_env` | same name |
| `airsenal/framework/fpl_team_utils.py` | `get_overall_points` | removed | removed: Alexa-era helper, only used by aws_utils |
| `airsenal/framework/fpl_team_utils.py` | `get_overall_ranking` | removed | removed: Alexa-era helper, only used by aws_utils |
| `airsenal/framework/fpl_team_utils.py` | `get_league_standings` | removed | removed: Alexa-era helper, only used by aws_utils |
| `airsenal/framework/multiprocessing_utils.py` | `set_multiprocessing_start_method` | `core/concurrency.py::set_multiprocessing_start_method` | same name |
| `airsenal/framework/multiprocessing_utils.py` | `SharedCounter` | `core/concurrency.py::SharedCounter` | same name |
| `airsenal/framework/multiprocessing_utils.py` | `SharedCounter.increment` | `core/concurrency.py::SharedCounter.increment` | same name |
| `airsenal/framework/multiprocessing_utils.py` | `SharedCounter.value` | `core/concurrency.py::SharedCounter.value` | same name |
| `airsenal/framework/multiprocessing_utils.py` | `CustomQueue` | `core/concurrency.py::CustomQueue` | same name |
| `airsenal/framework/multiprocessing_utils.py` | `CustomQueue.put` | `core/concurrency.py::CustomQueue.put` | same name |
| `airsenal/framework/multiprocessing_utils.py` | `CustomQueue.get` | `core/concurrency.py::CustomQueue.get` | same name |
| `airsenal/framework/multiprocessing_utils.py` | `CustomQueue.qsize` | `core/concurrency.py::CustomQueue.qsize` | same name |
| `airsenal/framework/multiprocessing_utils.py` | `CustomQueue.empty` | `core/concurrency.py::CustomQueue.empty` | same name |
| `airsenal/framework/optimization_squad.py` | `SquadOpt` | `optimization/squad_optimizers/genetic_algorithm.py::SquadOpt` | same name |
| `airsenal/framework/optimization_squad.py` | `SquadOpt.optimize` | `optimization/squad_optimizers/genetic_algorithm.py::SquadOpt.optimize` | same name |
| `airsenal/framework/optimization_squad.py` | `make_new_squad` | `optimization/squad_optimizers/genetic_algorithm.py::make_new_squad` | same name |
| `airsenal/framework/optimization_transfers.py` | `make_optimum_single_transfer` | `optimization/strategies/single.py::make_optimum_single_transfer` | same name |
| `airsenal/framework/optimization_transfers.py` | `make_optimum_double_transfer` | `optimization/strategies/double.py::make_optimum_double_transfer` | same name |
| `airsenal/framework/optimization_transfers.py` | `make_random_transfers` | `optimization/strategies/random_search.py::make_random_transfers` | same name |
| `airsenal/framework/optimization_transfers.py` | `make_best_transfers` | `optimization/transfer_optimizers/branches.py::make_best_transfers` | same name, shared by the transfer optimizers |
| `airsenal/framework/optimization_utils.py` | `check_tag_valid` | `db/queries/tags.py::check_tag_valid` | same name |
| `airsenal/framework/optimization_utils.py` | `calc_points_hit` | `optimization/moves.py::calc_points_hit` | same name |
| `airsenal/framework/optimization_utils.py` | `calc_free_transfers` | `optimization/moves.py::calc_free_transfers` | same name |
| `airsenal/framework/optimization_utils.py` | `get_starting_squad` | `squad/history.py::get_starting_squad` | same name |
| `airsenal/framework/optimization_utils.py` | `get_squad_from_transactions` | `squad/history.py::get_squad_from_transactions` | same name |
| `airsenal/framework/optimization_utils.py` | `get_discounted_squad_score` | `optimization/squad_score.py::get_discounted_squad_score` | same name |
| `airsenal/framework/optimization_utils.py` | `get_baseline_strat` | `optimization/plan.py::baseline_plan` | renamed (Strategy -> Plan) |
| `airsenal/framework/optimization_utils.py` | `fill_suggestion_table` | `optimization/persist.py::fill_suggestion_table` | same name |
| `airsenal/framework/optimization_utils.py` | `fill_transaction_table` | `optimization/persist.py::fill_transaction_table` | same name |
| `airsenal/framework/optimization_utils.py` | `fill_initial_suggestion_table` | `optimization/persist.py::fill_initial_suggestion_table` | same name |
| `airsenal/framework/optimization_utils.py` | `fill_initial_transaction_table` | `optimization/persist.py::fill_initial_transaction_table` | same name |
| `airsenal/framework/optimization_utils.py` | `strategy_involves_N_or_more_transfers_in_gw` | removed | removed: no callers |
| `airsenal/framework/optimization_utils.py` | `make_strategy_id` | removed | removed: no callers |
| `airsenal/framework/optimization_utils.py` | `get_num_increments` | `optimization/protocols.py::progress_total` | split into each strategy's num_increments, read via progress_total |
| `airsenal/framework/optimization_utils.py` | `next_week_transfers` | `optimization/transfer_optimizers/branches.py::next_gameweek_transfers` | renamed |
| `airsenal/framework/optimization_utils.py` | `count_expected_outputs` | `optimization/transfer_optimizers/branches.py::count_expected_outputs` | same name |
| `airsenal/framework/optimization_utils.py` | `get_discount_factor` | `optimization/squad_score.py::get_discount_factor` | same name |
| `airsenal/framework/player.py` | `CandidatePlayer` | `squad/player.py::CandidatePlayer` | same name |
| `airsenal/framework/player.py` | `CandidatePlayer.calc_predicted_points` | `squad/player.py::CandidatePlayer.calc_predicted_points` | same name |
| `airsenal/framework/player.py` | `CandidatePlayer.get_predicted_points` | removed | removed: only tests used it |
| `airsenal/framework/player.py` | `DummyPlayer` | `squad/player.py::DummyPlayer` | same name |
| `airsenal/framework/player.py` | `DummyPlayer.calc_predicted_points` | `squad/player.py::DummyPlayer.calc_predicted_points` | same name |
| `airsenal/framework/player.py` | `DummyPlayer.get_predicted_points` | removed | removed: only tests used it |
| `airsenal/framework/player_model.py` | `get_empirical_bayes_estimates` | `prediction/player_models/scaling.py::get_empirical_bayes_estimates` | same name |
| `airsenal/framework/player_model.py` | `scale_goals_by_minutes` | `prediction/player_models/scaling.py::scale_goals_by_minutes` | same name |
| `airsenal/framework/player_model.py` | `BasePlayerModel` | `prediction/protocols.py::PlayerModel` | ABC deleted; duplicated the PlayerModel protocol |
| `airsenal/framework/player_model.py` | `BasePlayerModel.fit` | `prediction/protocols.py::PlayerModel.fit` | now a protocol method |
| `airsenal/framework/player_model.py` | `BasePlayerModel.get_probs` | `prediction/protocols.py::PlayerModel.predict_involvement` | renamed; returns PlayerInvolvement |
| `airsenal/framework/player_model.py` | `BasePlayerModel.get_probs_for_player` | removed | removed: never called, dropped from contract |
| `airsenal/framework/player_model.py` | `NumpyroPlayerModel` | `prediction/player_models/mcmc.py::NumpyroPlayerModel` | same name |
| `airsenal/framework/player_model.py` | `NumpyroPlayerModel.fit` | `prediction/player_models/mcmc.py::NumpyroPlayerModel.fit` | same name |
| `airsenal/framework/player_model.py` | `NumpyroPlayerModel.get_probs` | `prediction/player_models/mcmc.py::NumpyroPlayerModel.predict_involvement` | renamed; returns PlayerInvolvement |
| `airsenal/framework/player_model.py` | `NumpyroPlayerModel.get_probs_for_player` | removed | removed: never called, dropped from contract |
| `airsenal/framework/player_model.py` | `ConjugatePlayerModel` | `prediction/player_models/conjugate.py::ConjugatePlayerModel` | same name |
| `airsenal/framework/player_model.py` | `ConjugatePlayerModel.fit` | `prediction/player_models/conjugate.py::ConjugatePlayerModel.fit` | same name |
| `airsenal/framework/player_model.py` | `ConjugatePlayerModel.get_prior` | `prediction/player_models/conjugate.py::ConjugatePlayerModel.get_prior` | same name |
| `airsenal/framework/player_model.py` | `ConjugatePlayerModel.get_posterior` | `prediction/player_models/conjugate.py::dirichlet_update` | inlined into shared Dirichlet update helper |
| `airsenal/framework/player_model.py` | `ConjugatePlayerModel.get_probs` | `prediction/player_models/conjugate.py::ConjugatePlayerModel.predict_involvement` | renamed; returns PlayerInvolvement |
| `airsenal/framework/player_model.py` | `ConjugatePlayerModel.get_probs_for_player` | removed | removed: never called, dropped from contract |
| `airsenal/framework/prediction_utils.py` | `check_absence` | removed | removed: zero callers; Absence table later removed |
| `airsenal/framework/prediction_utils.py` | `get_player_history_df` | `prediction/features.py::get_player_history_df` | same name |
| `airsenal/framework/prediction_utils.py` | `get_attacking_points` | `prediction/point_components/attacking.py::get_attacking_points` | same name |
| `airsenal/framework/prediction_utils.py` | `get_defending_points` | `prediction/point_components/defending.py::get_defending_points` | same name |
| `airsenal/framework/prediction_utils.py` | `get_bonus_points` | `prediction/point_components/bonus.py::BonusComponent.expected_points` | now a PointComponent method |
| `airsenal/framework/prediction_utils.py` | `get_def_con_points` | `prediction/point_components/def_con.py::DefConComponent.expected_points` | now a PointComponent method |
| `airsenal/framework/prediction_utils.py` | `get_save_points` | `prediction/point_components/saves.py::SaveComponent.expected_points` | now a PointComponent method |
| `airsenal/framework/prediction_utils.py` | `get_card_points` | `prediction/point_components/cards.py::CardComponent.expected_points` | now a PointComponent method |
| `airsenal/framework/prediction_utils.py` | `calc_predicted_points_for_player` | `prediction/points_models/component.py::ComponentPointsModel.predict` | now the points model; loop in prediction/run.py::make_prediction |
| `airsenal/framework/prediction_utils.py` | `calc_predicted_points_for_pos` | removed | removed: dead, superseded by per-player version |
| `airsenal/framework/prediction_utils.py` | `make_prediction` | `prediction/run.py::make_prediction` | same name |
| `airsenal/framework/prediction_utils.py` | `fill_ep` | removed | removed: dead since 2018, FPL expected-points loading dropped |
| `airsenal/framework/prediction_utils.py` | `process_player_data` | `prediction/features.py::process_player_data` | same name |
| `airsenal/framework/prediction_utils.py` | `fit_player_data` | `prediction/player_models/fitting.py::fit_player_data` | same name |
| `airsenal/framework/prediction_utils.py` | `get_all_fitted_player_data` | `prediction/player_models/fitting.py::get_all_fitted_player_data` | same name |
| `airsenal/framework/prediction_utils.py` | `get_player_scores` | `db/queries/scores.py::get_player_scores`<br>`db/queries/scores.py::get_player_scores`<br>`db/queries/scores.py::get_player_scores`<br>`db/queries/scores.py::get_player_scores` | same name, defined in several places |
| `airsenal/framework/prediction_utils.py` | `mean_group_prior` | `prediction/point_components/empirical_bayes.py::mean_group_prior` | same name |
| `airsenal/framework/prediction_utils.py` | `fit_bonus_points` | `prediction/point_components/bonus.py::fit_bonus_points` | same name |
| `airsenal/framework/prediction_utils.py` | `fit_save_points` | `prediction/point_components/saves.py::fit_save_points` | same name |
| `airsenal/framework/prediction_utils.py` | `fit_card_points` | `prediction/point_components/cards.py::fit_card_points` | same name |
| `airsenal/framework/prediction_utils.py` | `fit_def_con` | `prediction/point_components/def_con.py::fit_def_con` | same name |
| `airsenal/framework/random_team_model.py` | `RandomMatchPredictor` | `prediction/team_models/random_model.py::RandomTeamModel` | renamed |
| `airsenal/framework/random_team_model.py` | `RandomMatchPredictor.fit` | `prediction/team_models/random_model.py::RandomTeamModel.fit` | class renamed |
| `airsenal/framework/random_team_model.py` | `RandomMatchPredictor.predict_score_proba` | `prediction/team_models/random_model.py::RandomTeamModel.predict_score_n_proba` | replaced by per-team goal-count pmf |
| `airsenal/framework/random_team_model.py` | `RandomMatchPredictor.add_new_team` | `prediction/team_models/random_model.py::RandomTeamModel.add_new_team` | class renamed |
| `airsenal/framework/schema.py` | `Base` | `db/models.py::Base` | same name |
| `airsenal/framework/schema.py` | `Player` | `db/models.py::Player` | same name |
| `airsenal/framework/schema.py` | `Player.team` | `db/models.py::Player.team` | same name |
| `airsenal/framework/schema.py` | `Player.price` | `db/models.py::Player.price` | same name |
| `airsenal/framework/schema.py` | `Player.position` | `db/models.py::Player.position` | same name |
| `airsenal/framework/schema.py` | `Player.is_injured_or_suspended` | `db/models.py::Player.is_injured_or_suspended` | same name |
| `airsenal/framework/schema.py` | `Player.get_gameweek_attributes` | `db/models.py::Player.get_gameweek_attributes` | same name |
| `airsenal/framework/schema.py` | `PlayerMapping` | `db/models.py::PlayerMapping` | same name |
| `airsenal/framework/schema.py` | `PlayerAttributes` | `db/models.py::PlayerAttributes` | same name |
| `airsenal/framework/schema.py` | `Absence` | removed | removed: availability now on PlayerAttributes |
| `airsenal/framework/schema.py` | `Result` | `db/models.py::Result` | same name |
| `airsenal/framework/schema.py` | `Fixture` | `db/models.py::Fixture` | same name |
| `airsenal/framework/schema.py` | `PlayerScore` | `db/models.py::PlayerScore` | same name |
| `airsenal/framework/schema.py` | `PlayerPrediction` | `db/models.py::PlayerPrediction` | same name |
| `airsenal/framework/schema.py` | `Transaction` | `db/models.py::Transaction` | same name |
| `airsenal/framework/schema.py` | `TransferSuggestion` | `db/models.py::TransferSuggestion` | same name |
| `airsenal/framework/schema.py` | `FifaTeamRating` | `db/models.py::FifaTeamRating` | same name |
| `airsenal/framework/schema.py` | `Team` | `db/models.py::Team`<br>`remote/transfermarkt.py::Team` | same name, defined in several places |
| `airsenal/framework/schema.py` | `SessionSquad` | removed | removed: Flask API remnant, table unused |
| `airsenal/framework/schema.py` | `SessionBudget` | removed | removed: Flask API remnant, table unused |
| `airsenal/framework/schema.py` | `get_connection_string` | `db/engine.py::get_connection_string` | same name |
| `airsenal/framework/schema.py` | `get_session` | `db/session.py::get_session` | same name |
| `airsenal/framework/schema.py` | `session_scope` | `db/session.py::session_scope` | same name |
| `airsenal/framework/schema.py` | `clean_database` | `db/session.py::clean_database` | same name |
| `airsenal/framework/schema.py` | `database_is_empty` | `db/queries/teams.py::database_is_empty` | same name |
| `airsenal/framework/season.py` | `get_current_season` | `game/season.py::get_current_season` | same name |
| `airsenal/framework/season.py` | `get_teams_for_season` | `db/queries/teams.py::get_teams_for_season`<br>`remote/transfermarkt.py::get_teams_for_season` | same name, defined in several places |
| `airsenal/framework/season.py` | `season_str_to_year` | `game/season.py::season_str_to_year` | same name |
| `airsenal/framework/season.py` | `sort_seasons` | `game/season.py::sort_seasons` | same name |
| `airsenal/framework/squad.py` | `Squad` | `squad/squad.py::Squad` | same name |
| `airsenal/framework/squad.py` | `Squad.is_complete` | `squad/squad.py::Squad.is_complete` | same name |
| `airsenal/framework/squad.py` | `Squad.add_player` | `squad/squad.py::Squad.add_player` | same name |
| `airsenal/framework/squad.py` | `Squad.remove_player` | `squad/squad.py::Squad.remove_player` | same name |
| `airsenal/framework/squad.py` | `Squad.get_player_from_id` | `squad/squad.py::Squad.get_player_from_id` | same name |
| `airsenal/framework/squad.py` | `Squad.get_sell_price_for_player` | `squad/squad.py::Squad.get_sell_price_for_player` | same name |
| `airsenal/framework/squad.py` | `Squad.check_no_duplicate_player` | `squad/squad.py::Squad.check_no_duplicate_player` | same name |
| `airsenal/framework/squad.py` | `Squad.check_num_in_position` | `squad/squad.py::Squad.check_num_in_position` | same name |
| `airsenal/framework/squad.py` | `Squad.check_num_per_team` | `squad/squad.py::Squad.check_num_per_team` | same name |
| `airsenal/framework/squad.py` | `Squad.check_cost` | `squad/squad.py::Squad.check_cost` | same name |
| `airsenal/framework/squad.py` | `Squad.optimize_subs` | `squad/lineup.py::choose_starting_eleven` | renamed; free function over players |
| `airsenal/framework/squad.py` | `Squad.order_substitutes` | `squad/lineup.py::order_substitutes` | now a free function in lineup.py |
| `airsenal/framework/squad.py` | `Squad.apply_formation` | `squad/lineup.py::apply_formation` | now a free function in lineup.py |
| `airsenal/framework/squad.py` | `Squad.get_formation` | `squad/lineup.py::formation_of` | renamed; free function in lineup.py |
| `airsenal/framework/squad.py` | `Squad.is_substitution_allowed` | `squad/lineup.py::formation_after` | split into is_formation_legal and formation_after |
| `airsenal/framework/squad.py` | `Squad.total_points_for_starting_11` | `squad/squad.py::Squad.total_points_for_starting_11` | same name |
| `airsenal/framework/squad.py` | `Squad.total_points_for_subs` | `squad/squad.py::Squad.total_points_for_subs` | same name |
| `airsenal/framework/squad.py` | `Squad.optimize_lineup` | `squad/squad.py::Squad.optimize_lineup` | same name |
| `airsenal/framework/squad.py` | `Squad.get_expected_points` | `squad/squad.py::Squad.get_expected_points` | same name |
| `airsenal/framework/squad.py` | `Squad.pick_captains` | `squad/lineup.py::pick_captains` | now a free function in lineup.py |
| `airsenal/framework/squad.py` | `Squad.get_actual_points` | `squad/squad.py::Squad.get_actual_points` | same name |
| `airsenal/framework/squad.py` | `Squad.sale_value` | `squad/squad.py::Squad.sale_value` | same name |
| `airsenal/framework/squad.py` | `get_current_squad_from_api` | `squad/squad.py::get_current_squad_from_api` | same name |
| `airsenal/framework/transaction_utils.py` | `free_hit_used_in_gameweek` | `squad/state.py::chip_used_in_gameweek` | generalised to any chip |
| `airsenal/framework/transaction_utils.py` | `count_transactions` | `db/queries/transactions.py::count_transactions` | same name |
| `airsenal/framework/transaction_utils.py` | `transaction_exists` | `db/queries/transactions.py::transaction_exists` | same name |
| `airsenal/framework/transaction_utils.py` | `add_transaction` | `db/queries/transactions.py::add_transaction` | same name |
| `airsenal/framework/transaction_utils.py` | `fill_initial_squad` | `squad/history.py::record_initial_squad_transactions` | renamed to avoid name clash |
| `airsenal/framework/transaction_utils.py` | `update_squad` | `squad/history.py::update_squad` | same name |
| `airsenal/framework/utils.py` | `get_max_gameweek` | `db/queries/gameweeks.py::get_max_gameweek` | same name |
| `airsenal/framework/utils.py` | `get_next_gameweek` | `db/queries/gameweeks.py::next_gameweek` | renamed; single cached function |
| `airsenal/framework/utils.py` | `parse_datetime` | `core/dates.py::parse_datetime` | same name |
| `airsenal/framework/utils.py` | `parse_date` | `core/dates.py::parse_date` | same name |
| `airsenal/framework/utils.py` | `get_return_gameweek_by_date` | `db/queries/gameweeks.py::get_return_gameweek_by_date` | same name |
| `airsenal/framework/utils.py` | `get_gameweeks_array` | `db/queries/gameweeks.py::get_gameweeks_array` | same name |
| `airsenal/framework/utils.py` | `get_next_season` | `game/season.py::get_next_season` | same name |
| `airsenal/framework/utils.py` | `get_start_end_dates_of_season` | `remote/transfermarkt.py::get_start_end_dates_of_season` | same name |
| `airsenal/framework/utils.py` | `get_previous_season` | `game/season.py::get_previous_season` | same name |
| `airsenal/framework/utils.py` | `get_past_seasons` | `game/season.py::get_past_seasons` | same name |
| `airsenal/framework/utils.py` | `get_current_players` | removed | removed: dead, caller switched to API data |
| `airsenal/framework/utils.py` | `get_bank` | `squad/state.py::get_bank` | same name |
| `airsenal/framework/utils.py` | `get_entry_start_gameweek` | `squad/state.py::get_entry_start_gameweek` | same name |
| `airsenal/framework/utils.py` | `get_free_transfers` | `squad/state.py::get_free_transfers` | same name |
| `airsenal/framework/utils.py` | `get_gameweek_by_date` | `db/queries/gameweeks.py::get_gameweek_by_date` | same name |
| `airsenal/framework/utils.py` | `get_team_name` | `db/queries/teams.py::get_team_name` | same name |
| `airsenal/framework/utils.py` | `get_player` | `db/queries/players.py::get_player` | same name |
| `airsenal/framework/utils.py` | `get_player_from_api_id` | `db/queries/players.py::get_player_from_api_id` | same name |
| `airsenal/framework/utils.py` | `get_player_name` | `db/queries/players.py::get_player_name` | same name |
| `airsenal/framework/utils.py` | `get_player_id` | removed | removed: use db/queries/players.py::get_player |
| `airsenal/framework/utils.py` | `list_teams` | removed | removed: Flask API remnant |
| `airsenal/framework/utils.py` | `list_players` | `db/queries/players.py::list_players` | same name |
| `airsenal/framework/utils.py` | `is_future_gameweek` | `db/queries/gameweeks.py::is_future_gameweek` | same name |
| `airsenal/framework/utils.py` | `get_max_matches_per_player` | removed | removed: padding length now from loaded rows |
| `airsenal/framework/utils.py` | `get_player_attributes` | `db/queries/players.py::get_player_attributes` | same name |
| `airsenal/framework/utils.py` | `get_fixtures_for_player` | `db/queries/fixtures.py::get_fixtures_for_player` | same name |
| `airsenal/framework/utils.py` | `get_next_fixture_for_player` | removed | removed: Flask API remnant |
| `airsenal/framework/utils.py` | `get_fixtures_for_season` | `db/queries/fixtures.py::get_fixtures_for_season` | same name |
| `airsenal/framework/utils.py` | `get_fixtures_for_gameweek` | `db/queries/fixtures.py::get_fixtures_for_gameweeks` | renamed (plural), takes list of gameweeks |
| `airsenal/framework/utils.py` | `get_fixture_teams` | `db/queries/fixtures.py::get_fixture_teams`<br>`export/player_details.py::get_fixture_teams` | same name, defined in several places |
| `airsenal/framework/utils.py` | `get_player_scores` | `db/queries/scores.py::get_player_scores`<br>`db/queries/scores.py::get_player_scores`<br>`db/queries/scores.py::get_player_scores`<br>`db/queries/scores.py::get_player_scores` | same name, defined in several places |
| `airsenal/framework/utils.py` | `get_players_for_gameweek` | `squad/state.py::get_players_for_gameweek` | same name |
| `airsenal/framework/utils.py` | `get_previous_points_for_same_fixture` | removed | removed: never called |
| `airsenal/framework/utils.py` | `get_predicted_points_for_player` | `db/queries/predictions.py::get_predicted_points_for_player` | same name |
| `airsenal/framework/utils.py` | `get_predicted_points` | `db/queries/predictions.py::get_predicted_points` | same name |
| `airsenal/framework/utils.py` | `get_top_predicted_points` | `reporting/top_players.py::get_top_predicted_points` | same name |
| `airsenal/framework/utils.py` | `predicted_points_discord_payload` | `reporting/top_players.py::predicted_points_discord_payload` | same name |
| `airsenal/framework/utils.py` | `get_return_gameweek_from_news` | `ingest/player_attributes.py::get_return_gameweek_from_news` | same name |
| `airsenal/framework/utils.py` | `calc_average_minutes` | `prediction/minutes.py::calc_average_minutes` | same name |
| `airsenal/framework/utils.py` | `estimate_minutes_from_prev_season` | `prediction/minutes.py::estimate_minutes_from_prev_season` | same name |
| `airsenal/framework/utils.py` | `get_recent_playerscore_rows` | `db/queries/scores.py::get_recent_playerscore_rows` | same name |
| `airsenal/framework/utils.py` | `get_playerscores_for_player_gameweek` | `db/queries/scores.py::get_playerscores_for_player_gameweek` | same name |
| `airsenal/framework/utils.py` | `get_recent_scores_for_player` | removed | removed: Flask API remnant |
| `airsenal/framework/utils.py` | `get_recent_minutes_for_player` | `prediction/minutes.py::get_recent_minutes_for_player` | same name |
| `airsenal/framework/utils.py` | `was_historic_absence` | `db/models.py::Player.is_injured_or_suspended` | removed; availability now read from PlayerAttributes |
| `airsenal/framework/utils.py` | `get_last_complete_gameweek_in_db` | `db/queries/gameweeks.py::get_last_complete_gameweek_in_db` | same name |
| `airsenal/framework/utils.py` | `get_last_finished_gameweek` | `remote/fpl_api.py::FPLDataFetcher.get_last_finished_gameweek` | now a fetcher method |
| `airsenal/framework/utils.py` | `get_latest_prediction_tag` | `db/queries/tags.py::get_latest_prediction_tag` | same name |
| `airsenal/framework/utils.py` | `get_latest_fixture_tag` | `db/queries/tags.py::get_latest_fixture_tag` | same name |
| `airsenal/framework/utils.py` | `find_fixture` | `db/queries/fixtures.py::find_fixture` | same name |
| `airsenal/framework/utils.py` | `get_player_team_from_fixture` | `db/queries/fixtures.py::get_player_team_from_fixture` | same name |
| `airsenal/framework/utils.py` | `is_transfer_deadline_today` | removed | removed: never called |
| `airsenal/framework/utils.py` | `fastcopy` | `core/copy.py::fastcopy` | same name |
| `airsenal/scripts/airsenal_run_pipeline.py` | `run_pipeline` | `pipeline/run.py::AIrsenalPipeline.run` | replaced by AIrsenalPipeline; CLI: airsenal run |
| `airsenal/scripts/airsenal_run_pipeline.py` | `setup_database` | `pipeline/run.py::AIrsenalPipeline._refresh_database` | now a private pipeline method |
| `airsenal/scripts/airsenal_run_pipeline.py` | `setup_chips` | `optimization/moves.py::ChipGameweeks` | chip dict replaced by typed ChipGameweeks |
| `airsenal/scripts/airsenal_run_pipeline.py` | `update_database` | `ingest/update.py::update_database` | same name |
| `airsenal/scripts/airsenal_run_pipeline.py` | `run_prediction` | `pipeline/run.py::AIrsenalPipeline.predict` | now a pipeline method |
| `airsenal/scripts/airsenal_run_pipeline.py` | `run_make_squad` | `pipeline/run.py::AIrsenalPipeline.optimize` | merged into optimize (new-squad branch) |
| `airsenal/scripts/airsenal_run_pipeline.py` | `run_optimize_squad` | `pipeline/run.py::AIrsenalPipeline.optimize` | merged into optimize (transfers branch) |
| `airsenal/scripts/airsenal_run_pipeline.py` | `set_starting_11` | `pipeline/run.py::AIrsenalPipeline._apply` | merged into _apply, calls set_lineup |
| `airsenal/scripts/data_sanity_checks.py` | `result_string` | `ingest/checks.py::result_string` | same name |
| `airsenal/scripts/data_sanity_checks.py` | `season_num_teams` | `ingest/checks.py::season_num_teams` | same name |
| `airsenal/scripts/data_sanity_checks.py` | `season_num_new_teams` | `ingest/checks.py::season_num_new_teams` | same name |
| `airsenal/scripts/data_sanity_checks.py` | `season_num_fixtures` | `ingest/checks.py::season_num_fixtures` | same name |
| `airsenal/scripts/data_sanity_checks.py` | `fixture_player_teams` | `ingest/checks.py::fixture_player_teams` | same name |
| `airsenal/scripts/data_sanity_checks.py` | `fixture_num_players` | `ingest/checks.py::fixture_num_players` | same name |
| `airsenal/scripts/data_sanity_checks.py` | `fixture_num_goals` | `ingest/checks.py::fixture_num_goals` | same name |
| `airsenal/scripts/data_sanity_checks.py` | `fixture_num_assists` | `ingest/checks.py::fixture_num_assists` | same name |
| `airsenal/scripts/data_sanity_checks.py` | `fixture_num_conceded` | `ingest/checks.py::fixture_num_conceded` | same name |
| `airsenal/scripts/data_sanity_checks.py` | `run_all_checks` | `ingest/checks.py::run_all_checks` | same name |
| `airsenal/scripts/dump_db_contents.py` | `save_table_fields` | `export/db_dump.py::save_table_fields` | same name |
| `airsenal/scripts/dump_db_contents.py` | `write_rows_to_csv` | `export/db_dump.py::write_rows_to_csv` | same name |
| `airsenal/scripts/duplicate_names.py` | `find_duplicate_names` | `tools/duplicate_names.py::find_duplicate_names` | moved to tools/ |
| `airsenal/scripts/fill_absence_table.py` | `load_absences` | `ingest/absences.py::get_availability_from_absences` | CSV now feeds PlayerAttributes availability |
| `airsenal/scripts/fill_absence_table.py` | `make_absence_table` | `ingest/player_attributes.py::fill_availability_for_season` | Absence table removed; availability written to attributes |
| `airsenal/scripts/fill_db_init.py` | `check_clean_db` | `ingest/init_db.py::check_clean_db` | same name |
| `airsenal/scripts/fill_db_init.py` | `make_init_db` | `ingest/init_db.py::make_init_db` | same name |
| `airsenal/scripts/fill_db_init.py` | `check_positive_int` | removed | removed: argparse validator; typer min=1 instead |
| `airsenal/scripts/fill_fifa_ratings_table.py` | `make_fifa_ratings_table` | `ingest/fifa_ratings.py::make_fifa_ratings_table` | same name |
| `airsenal/scripts/fill_fixture_table.py` | `fill_fixtures_from_file` | `ingest/fixtures.py::fill_fixtures_from_file` | same name |
| `airsenal/scripts/fill_fixture_table.py` | `fill_fixtures_from_api` | `ingest/fixtures.py::fill_fixtures_from_api` | same name |
| `airsenal/scripts/fill_fixture_table.py` | `make_fixture_table` | `ingest/fixtures.py::make_fixture_table` | same name |
| `airsenal/scripts/fill_player_attributes_table.py` | `fill_attributes_table_from_file` | `ingest/player_attributes.py::fill_attributes_table_from_file` | same name |
| `airsenal/scripts/fill_player_attributes_table.py` | `fill_attributes_table_from_api` | `ingest/player_attributes.py::fill_attributes_table_from_api` | same name |
| `airsenal/scripts/fill_player_attributes_table.py` | `make_attributes_table` | `ingest/player_attributes.py::make_attributes_table` | same name |
| `airsenal/scripts/fill_player_mappings_table.py` | `load_mappings_data` | `ingest/player_mappings.py::load_mappings_data` | same name |
| `airsenal/scripts/fill_player_mappings_table.py` | `add_mappings` | `ingest/player_mappings.py::add_mappings` | same name |
| `airsenal/scripts/fill_player_mappings_table.py` | `make_player_mappings_table` | `ingest/player_mappings.py::make_player_mappings_table` | same name |
| `airsenal/scripts/fill_player_table.py` | `find_player_in_table` | `ingest/players.py::find_player_in_table` | same name |
| `airsenal/scripts/fill_player_table.py` | `num_players_in_table` | removed | removed: ids come from autoincrement |
| `airsenal/scripts/fill_player_table.py` | `fill_player_table_from_file` | `ingest/players.py::fill_player_table_from_file` | same name |
| `airsenal/scripts/fill_player_table.py` | `fill_player_table_from_api` | `ingest/players.py::fill_player_table_from_api` | same name |
| `airsenal/scripts/fill_player_table.py` | `make_init_player_table` | `ingest/players.py::make_init_player_table` | same name |
| `airsenal/scripts/fill_player_table.py` | `make_remaining_player_table` | `ingest/players.py::make_remaining_player_table` | same name |
| `airsenal/scripts/fill_player_table.py` | `make_player_table` | `ingest/players.py::make_player_table` | same name |
| `airsenal/scripts/fill_playerscore_table.py` | `download_with_resume` | `remote/download.py::download_with_resume` | same name |
| `airsenal/scripts/fill_playerscore_table.py` | `load_attributes_history` | `ingest/attributes_history.py::load_attributes_history` | same name |
| `airsenal/scripts/fill_playerscore_table.py` | `get_status_from_attributes_history` | `ingest/player_scores.py::get_status_from_attributes_history` | same name |
| `airsenal/scripts/fill_playerscore_table.py` | `fill_playerscores_from_json` | `ingest/player_scores.py::fill_playerscores_from_json` | same name |
| `airsenal/scripts/fill_playerscore_table.py` | `fill_playerscores_from_api` | `ingest/player_scores.py::fill_playerscores_from_api` | same name |
| `airsenal/scripts/fill_playerscore_table.py` | `make_playerscore_table` | `ingest/player_scores.py::make_playerscore_table` | same name |
| `airsenal/scripts/fill_predictedscore_table.py` | `calc_all_predicted_points` | `prediction/run.py::calc_all_predicted_points` | same name |
| `airsenal/scripts/fill_predictedscore_table.py` | `make_predictedscore_table` | `prediction/run.py::make_predictedscore_table` | same name |
| `airsenal/scripts/fill_result_table.py` | `fill_results_from_csv` | `ingest/results.py::fill_results_from_csv` | same name |
| `airsenal/scripts/fill_result_table.py` | `fill_results_from_api` | `ingest/results.py::fill_results_from_api` | same name |
| `airsenal/scripts/fill_result_table.py` | `make_result_table` | `ingest/results.py::make_result_table` | same name |
| `airsenal/scripts/fill_team_table.py` | `fill_team_table_from_file` | `ingest/teams.py::fill_team_table_from_file` | same name |
| `airsenal/scripts/fill_team_table.py` | `make_team_table` | `ingest/teams.py::make_team_table` | same name |
| `airsenal/scripts/fill_transfersuggestion_table.py` | `optimize` | `optimization/transfer_optimizers/tree_search.py::optimize` | same name |
| `airsenal/scripts/fill_transfersuggestion_table.py` | `find_best_strat_from_json` | removed | removed: plans returned on a queue, not JSON files |
| `airsenal/scripts/fill_transfersuggestion_table.py` | `save_baseline_score` | removed | removed: plans returned on a queue, not JSON files |
| `airsenal/scripts/fill_transfersuggestion_table.py` | `find_baseline_score_from_json` | removed | removed: plans returned on a queue, not JSON files |
| `airsenal/scripts/fill_transfersuggestion_table.py` | `print_strat` | `reporting/optimization.py::print_plan_table` | replaced by renderers in reporting/optimization.py |
| `airsenal/scripts/fill_transfersuggestion_table.py` | `discord_payload` | `reporting/optimization.py::discord_payload` | same name |
| `airsenal/scripts/fill_transfersuggestion_table.py` | `print_team_for_next_gw` | `optimization/run_transfers.py::squad_for_next_gameweek` | split: squad_for_next_gameweek + reporting formation_table |
| `airsenal/scripts/fill_transfersuggestion_table.py` | `run_optimization` | `optimization/run_transfers.py::run_optimization` | same name |
| `airsenal/scripts/fill_transfersuggestion_table.py` | `construct_chip_dict` | `optimization/moves.py::ChipSchedule.from_gameweeks` | chip dict replaced by typed ChipSchedule |
| `airsenal/scripts/fill_transfersuggestion_table.py` | `sanity_check_args` | `cli/optimize.py::_check_gameweek_args` | moved to CLI; one check dropped |
| `airsenal/scripts/get_transfer_suggestions.py` | `get_transfer_suggestions` | `db/queries/predictions.py::get_transfer_suggestions` | same name |
| `airsenal/scripts/get_transfer_suggestions.py` | `build_strategy_string` | removed | removed: only reachable from module's __main__ |
| `airsenal/scripts/make_player_details.py` | `make_player_details` | `export/player_details.py::make_player_details` | same name |
| `airsenal/scripts/make_player_details.py` | `get_team_mapping` | `export/player_details.py::get_team_mapping` | same name |
| `airsenal/scripts/make_player_details.py` | `get_fixture_teams` | `db/queries/fixtures.py::get_fixture_teams`<br>`export/player_details.py::get_fixture_teams` | same name, defined in several places |
| `airsenal/scripts/make_player_details.py` | `get_played_for` | `export/player_details.py::get_played_for` | same name |
| `airsenal/scripts/make_player_history_table.py` | `get_player_history_table` | removed | removed: superseded by prediction/features.py::get_player_history_df |
| `airsenal/scripts/make_player_summary.py` | `make_player_summary` | `export/player_summary.py::make_player_summary` | same name |
| `airsenal/scripts/make_results.py` | `make_results` | `export/results.py::make_results` | same name |
| `airsenal/scripts/make_transfers.py` | `check_proceed` | `apply/lineup.py::check_proceed`<br>`apply/transfers.py::check_proceed` | same name, defined in several places |
| `airsenal/scripts/make_transfers.py` | `deduct_transfer_price` | `apply/transfers.py::bank_after_transfers` | renamed |
| `airsenal/scripts/make_transfers.py` | `print_output` | `apply/transfers.py::print_output` | same name |
| `airsenal/scripts/make_transfers.py` | `get_sell_price` | `apply/transfers.py::get_sell_price` | same name |
| `airsenal/scripts/make_transfers.py` | `get_gw_transfer_suggestions` | `apply/transfers.py::get_suggested_transfers` | renamed; returns SuggestedTransfers |
| `airsenal/scripts/make_transfers.py` | `price_transfers` | `apply/transfers.py::price_transfers` | same name |
| `airsenal/scripts/make_transfers.py` | `separate_transfers_in_or_out` | `apply/transfers.py::separate_transfers_in_or_out` | same name |
| `airsenal/scripts/make_transfers.py` | `sort_by_position` | `apply/transfers.py::pair_by_position` | replaced; sorted_by_position wraps it |
| `airsenal/scripts/make_transfers.py` | `remove_duplicates` | `apply/transfers.py::remove_duplicates` | same name |
| `airsenal/scripts/make_transfers.py` | `build_init_priced_transfers` | `apply/transfers.py::build_init_priced_transfers` | same name |
| `airsenal/scripts/make_transfers.py` | `build_transfer_payload` | `apply/transfers.py::build_transfer_payload` | same name |
| `airsenal/scripts/make_transfers.py` | `make_transfers` | `apply/transfers.py::make_transfers` | same name |
| `airsenal/scripts/match_player_names.py` | `find_best_match` | removed | removed: name-matching one-off tool dropped |
| `airsenal/scripts/match_team_names.py` | `find_best_match` | removed | removed: name-matching one-off tool dropped |
| `airsenal/scripts/plot_league_standings.py` | `get_team_ids` | `reporting/plots.py::plot_standings` | inlined into plot_standings; CLI: airsenal plot |
| `airsenal/scripts/plot_league_standings.py` | `get_team_names` | `reporting/plots.py::plot_standings` | inlined into plot_standings; CLI: airsenal plot |
| `airsenal/scripts/plot_league_standings.py` | `get_team_history` | `reporting/plots.py::get_team_history` | same name |
| `airsenal/scripts/replay_season.py` | `get_dummy_id` | `pipeline/replay.py::get_dummy_id` | same name |
| `airsenal/scripts/replay_season.py` | `print_replay_params` | `pipeline/replay.py::print_replay_params` | same name |
| `airsenal/scripts/replay_season.py` | `replay_season` | `pipeline/replay.py::replay_season` | same name |
| `airsenal/scripts/save_attributes.py` | `get_return_gameweek_by_date` | `db/queries/gameweeks.py::get_return_gameweek_by_date` | same name |
| `airsenal/scripts/save_attributes.py` | `season_is_active` | `export/attributes.py::season_is_active` | same name |
| `airsenal/scripts/save_attributes.py` | `save_attributes_from_api` | `export/attributes.py::save_attributes_from_api` | same name |
| `airsenal/scripts/save_expected_absences.py` | `save_absences` | removed | removed: absences exporter retired with Absence table |
| `airsenal/scripts/save_expected_absences.py` | `player_attribute_to_absence` | removed | removed: absences exporter retired with Absence table |
| `airsenal/scripts/scrape_transfermarkt.py` | `get_teams_for_season` | `db/queries/teams.py::get_teams_for_season`<br>`remote/transfermarkt.py::get_teams_for_season` | same name, defined in several places |
| `airsenal/scripts/scrape_transfermarkt.py` | `get_team_players` | `remote/transfermarkt.py::get_team_players` | same name |
| `airsenal/scripts/scrape_transfermarkt.py` | `tidy_df` | `remote/transfermarkt.py::tidy_df` | same name |
| `airsenal/scripts/scrape_transfermarkt.py` | `filter_season` | `remote/transfermarkt.py::filter_season` | same name |
| `airsenal/scripts/scrape_transfermarkt.py` | `get_player_injuries` | `remote/transfermarkt.py::get_player_injuries` | same name |
| `airsenal/scripts/scrape_transfermarkt.py` | `get_reason` | `remote/transfermarkt.py::get_reason` | same name |
| `airsenal/scripts/scrape_transfermarkt.py` | `get_player_suspensions` | `remote/transfermarkt.py::get_player_suspensions` | same name |
| `airsenal/scripts/scrape_transfermarkt.py` | `get_players_for_season` | `remote/transfermarkt.py::get_players_for_season` | same name |
| `airsenal/scripts/scrape_transfermarkt.py` | `remove_youth_or_reserve_suffix` | `remote/transfermarkt.py::remove_youth_or_reserve_suffix` | same name |
| `airsenal/scripts/scrape_transfermarkt.py` | `get_player_transfers` | `remote/transfermarkt.py::get_player_transfers` | same name |
| `airsenal/scripts/scrape_transfermarkt.py` | `get_player_team_history` | `remote/transfermarkt.py::get_player_team_history` | same name |
| `airsenal/scripts/scrape_transfermarkt.py` | `get_player_transfer_unavailability` | `remote/transfermarkt.py::get_player_transfer_unavailability` | same name |
| `airsenal/scripts/scrape_transfermarkt.py` | `get_season_absences` | `remote/transfermarkt.py::get_season_absences` | same name |
| `airsenal/scripts/scrape_transfermarkt.py` | `scrape_transfermarkt` | `remote/transfermarkt.py::scrape_transfermarkt` | same name |
| `airsenal/scripts/scrape_understat.py` | `get_matches_info` | removed | removed: understat scraper had no importers |
| `airsenal/scripts/scrape_understat.py` | `parse_match` | removed | removed: understat scraper had no importers |
| `airsenal/scripts/scrape_understat.py` | `get_season_info` | removed | removed: understat scraper had no importers |
| `airsenal/scripts/set_env.py` | `redact_db_password` | `cli/env.py::redact_db_password` | same name |
| `airsenal/scripts/set_env.py` | `print_env` | `cli/env.py::print_env` | same name |
| `airsenal/scripts/set_lineup.py` | `check_proceed` | `apply/lineup.py::check_proceed`<br>`apply/transfers.py::check_proceed` | same name, defined in several places |
| `airsenal/scripts/set_lineup.py` | `build_lineup_payload` | `apply/lineup.py::build_lineup_payload` | same name |
| `airsenal/scripts/set_lineup.py` | `get_lineup_from_payload` | `apply/lineup.py::get_lineup_from_payload` | same name |
| `airsenal/scripts/set_lineup.py` | `make_squad_transfers` | removed | removed: dead since 2021 and broken |
| `airsenal/scripts/set_lineup.py` | `set_lineup` | `apply/lineup.py::set_lineup` | same name |
| `airsenal/scripts/squad_builder.py` | `fill_initial_squad` | `optimization/run_squad.py::build_new_squad` | renamed; CLI: airsenal optimize squad |
| `airsenal/scripts/tune_player_time_weighting.py` | `EpsilonResult` | `tools/tune_player_time_weighting.py::ParameterResult` | renamed; moved to tools/ |
| `airsenal/scripts/tune_player_time_weighting.py` | `get_player_outcome_prob` | `prediction/evaluation.py::player_outcome_probability` | renamed; moved into evaluation module |
| `airsenal/scripts/tune_player_time_weighting.py` | `evaluate_params` | `tools/tune_player_time_weighting.py::evaluate_params` | moved to tools/, rewritten over evaluation.py |
| `airsenal/scripts/tune_team_time_weighting.py` | `EpsilonResult` | `tools/tune_team_time_weighting.py::EpsilonResult` | moved to tools/ |
| `airsenal/scripts/tune_team_time_weighting.py` | `evaluate_epsilon` | `tools/tune_team_time_weighting.py::evaluate_epsilon` | moved to tools/, rewritten over evaluation.py |
| `airsenal/scripts/update_db.py` | `update_transactions` | `ingest/update.py::update_transactions` | same name |
| `airsenal/scripts/update_db.py` | `update_results` | `ingest/update.py::update_results` | same name |
| `airsenal/scripts/update_db.py` | `update_players` | `ingest/update.py::update_players` | same name |
| `airsenal/scripts/update_db.py` | `add_players_to_db` | `ingest/update.py::add_players_to_db` | same name |
| `airsenal/scripts/update_db.py` | `update_attributes` | `ingest/update.py::update_attributes` | same name |
| `airsenal/scripts/update_db.py` | `update_db` | `ingest/update.py::update_db` | same name |
