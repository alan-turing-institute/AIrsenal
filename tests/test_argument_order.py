"""
The argument order in CodingConventions.md, ratcheted.

The convention is `other args -> player -> position -> team -> tag -> gameweek ->
season -> fpl_team_id -> dbsession -> fetcher -> verbose`. Unlike the naming
rules in `test_naming_conventions.py` it was never machine-checked, and it drifted:
53 functions do not follow it.

This is a ratchet, in the same spirit as the complexity limits in
`[tool.ruff.lint.pylint]`. `KNOWN_OFFENDERS` is today's list and nothing may be
added to it - a new function, or a renamed one, has to follow the convention.
Fixing one means deleting its line. Reordering a signature is not free (callers
passing positionally break), which is why they are not all fixed here.

Ordering the whole codebase to the convention was measured before writing this:
no ordering of these names fits the code well - the best possible one still
leaves 44 functions out of order - so there is no competing convention hiding
here to adopt instead. The documented order stands; the code has to move to it.
"""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "airsenal"

# The documented order. A function taking two or more of these must take them in
# this relative order; anything not named here is unconstrained.
ARGUMENT_ORDER = [
    "player",
    "player_id",
    "position",
    "team",
    "tag",
    "gameweek",
    "season",
    "fpl_team_id",
    "dbsession",
    "fetcher",
    "verbose",
]
RANK = {name: index for index, name in enumerate(ARGUMENT_ORDER)}

# "module.py:function" for every function that predates the check. Only ever
# shrinks.
KNOWN_OFFENDERS = {
    "apply/transfers.py:build_init_priced_transfers",
    "cli/optimize.py:transfers",
    "db/models.py:get_gameweek_attributes",
    "db/models.py:price",
    "db/models.py:team",
    "db/queries/gameweeks.py:is_future_gameweek",
    "db/queries/gameweeks.py:set",
    "db/queries/players.py:_warn_incomplete_data",
    "db/queries/players.py:get_max_matches_per_player",
    "db/queries/players.py:get_player_attributes",
    "db/queries/players.py:list_players",
    "db/queries/predictions.py:get_predicted_points",
    "db/queries/predictions.py:get_transfer_suggestions",
    "db/queries/scores.py:get_player_scores_df",
    "db/queries/transactions.py:add_transaction",
    "db/queries/transactions.py:transaction_exists",
    "optimization/persist.py:_add_transactions",
    "optimization/persist.py:_buy_prices",
    "optimization/persist.py:fill_initial_suggestion_table",
    "optimization/persist.py:fill_initial_transaction_table",
    "optimization/persist.py:fill_transaction_table",
    "optimization/run_transfers.py:transfer_rows",
    "pipeline/replay.py:_gameweek_outcome",
    "prediction/features.py:get_player_history_df",
    "prediction/features.py:process_player_data",
    "prediction/minutes.py:estimate_minutes_from_prev_season",
    "prediction/player_models/fitting.py:fit_player_data",
    "prediction/player_models/fitting.py:get_all_fitted_player_data",
    "prediction/points.py:calc_predicted_points_for_player",
    "prediction/points.py:get_save_points",
    "prediction/run.py:calc_all_predicted_points",
    "prediction/team_models/fitting.py:get_fitted_team_model",
    "prediction/team_models/fitting.py:get_result_dict",
    "prediction/team_models/fitting.py:get_training_data",
    "reporting/top_players.py:get_top_predicted_points",
    "squad/history.py:get_starting_squad",
    "squad/history.py:record_initial_squad_transactions",
    "squad/history.py:update_squad",
    "squad/lineup.py:choose_starting_eleven",
    "squad/lineup.py:order_substitutes",
    "squad/lineup.py:pick_captains",
    "squad/player.py:__init__",
    "squad/player.py:get_predicted_points",
    "squad/pricing.py:sell_price",
    "squad/squad.py:get_expected_points",
    "squad/squad.py:optimize_lineup",
    "squad/squad.py:order_substitutes",
    "squad/squad.py:total_points_for_starting_11",
    "squad/squad.py:total_points_for_subs",
    "squad/state.py:get_bank",
    "squad/state.py:get_free_transfers",
}


def source_files():
    return sorted(SRC.rglob("*.py"))


def out_of_order(path):
    """(qualified name, the convention args it takes) for each bad signature."""
    relative = str(path.relative_to(SRC))
    bad = []
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        args = [
            a.arg
            for a in node.args.args + node.args.kwonlyargs
            if a.arg not in ("self", "cls")
        ]
        known = [a for a in args if a in RANK]
        if len(known) < 2:
            continue
        if [RANK[a] for a in known] != sorted(RANK[a] for a in known):
            bad.append((f"{relative}:{node.name}", known))
    return bad


@pytest.mark.parametrize("path", source_files(), ids=lambda p: str(p.relative_to(SRC)))
def test_no_new_argument_order_offenders(path):
    """A function not already on the list takes its arguments in order."""
    correct = " -> ".join(ARGUMENT_ORDER)
    offenders = [
        f"{name} takes ({', '.join(args)}) - the order is {correct}"
        for name, args in out_of_order(path)
        if name not in KNOWN_OFFENDERS
    ]
    assert not offenders, "\n".join(offenders)


def test_the_offender_list_has_no_stale_entries():
    """
    Every name on the list is still a real, still-misordered function.

    Without this the list would quietly outlive the problems it records, and a
    renamed function would take its exemption with it.
    """
    actual = {name for path in source_files() for name, _ in out_of_order(path)}
    stale = sorted(KNOWN_OFFENDERS - actual)
    assert not stale, (
        "These are no longer out of order - delete them from KNOWN_OFFENDERS:\n"
        + "\n".join(stale)
    )
