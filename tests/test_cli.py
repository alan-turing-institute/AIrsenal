import pytest
from typer.testing import CliRunner

from airsenal.cli.main import app
from airsenal.core.lookup import ConfigError
from airsenal.prediction.player_models import (
    PLAYER_MODELS,
    build_player_model,
)
from airsenal.prediction.team_models import (
    TEAM_MODELS,
    build_team_model,
)

# Rich wraps help output to the terminal width, and a long option name wraps
# mid-word - so ask for a terminal wide enough that a flag stays one token.
runner = CliRunner(env={"COLUMNS": "200"})


def test_run_help():
    runner = CliRunner()

    result = runner.invoke(app, ["run", "--help"])

    assert result.exit_code == 0
    assert "Run the full AIrsenal pipeline." in result.stdout
    assert "--weeks-ahead" in result.stdout


def test_db_help():
    runner = CliRunner()

    result = runner.invoke(app, ["db", "--help"])

    assert result.exit_code == 0
    assert "create" in result.stdout
    assert "update" in result.stdout


def test_predict_help():
    runner = CliRunner()

    result = runner.invoke(app, ["predict", "--help"])

    assert result.exit_code == 0
    assert "Predict player scores for a gameweek range." in result.stdout
    assert "--team-model" in result.stdout


def test_optimize_help():
    runner = CliRunner()

    result = runner.invoke(app, ["optimize", "--help"])

    assert result.exit_code == 0
    assert "transfers" in result.stdout
    assert "squad" in result.stdout


def test_env_help():
    runner = CliRunner()

    result = runner.invoke(app, ["env", "--help"])

    assert result.exit_code == 0
    assert "delete" in result.stdout
    assert "names" in result.stdout


def test_apply_help():
    runner = CliRunner()

    result = runner.invoke(app, ["apply", "--help"])

    assert result.exit_code == 0
    assert "transfers" in result.stdout
    assert "lineup" in result.stdout


def test_dump_help():
    runner = CliRunner()

    result = runner.invoke(app, ["dump", "--help"])

    assert result.exit_code == 0
    assert "transfermarkt" in result.stdout
    assert "absences" in result.stdout


def test_replay_help():
    runner = CliRunner()

    result = runner.invoke(app, ["replay", "--help"])

    assert result.exit_code == 0
    assert "Replay a historical FPL season." in result.stdout
    assert "--resume" in result.stdout


def test_plot_help():
    runner = CliRunner()

    result = runner.invoke(app, ["plot", "--help"])

    assert result.exit_code == 0
    assert "Plot a mini-league metric by gameweek." in result.stdout


class TestModelSelection:
    """
    The model tables reach the user through these options.

    Before them, the only model choice was a `--sampling` boolean, and the
    hyperparameters it implied were silently dropped.
    """

    @pytest.mark.parametrize(
        ("command", "option"),
        [
            (["predict"], "--player-model"),
            (["predict"], "--team-model"),
            (["predict"], "--epsilon"),
            (["run"], "--player-model"),
            (["run"], "--team-model"),
            (["run"], "--epsilon"),
            (["replay"], "--player-model"),
            (["replay"], "--team-model"),
            (["replay"], "--epsilon"),
            (["optimize", "squad"], "--num-generations"),
            (["optimize", "squad"], "--population-size"),
            (["optimize", "transfers"], "--save-plans"),
            (["optimize", "transfers"], "--num-iterations"),
        ],
    )
    def test_option_is_offered(self, command, option):
        result = runner.invoke(app, [*command, "--help"])
        assert result.exit_code == 0
        assert option in _flatten(result.stdout)

    def test_registered_model_names_appear_in_the_help(self):
        result = runner.invoke(app, ["predict", "--help"])
        assert result.exit_code == 0
        text = _flatten(result.stdout)
        for name in (*PLAYER_MODELS, *TEAM_MODELS):
            assert name in text

    def test_unknown_player_model_lists_the_available_ones(self):
        with pytest.raises(ConfigError, match="Choose from: conjugate, constant"):
            build_player_model("nope")

    def test_unknown_team_model_lists_the_available_ones(self):
        with pytest.raises(ConfigError, match="Choose from: constant, extended"):
            build_team_model("nope")


ALL_COMMANDS = [
    ["run"],
    ["predict"],
    ["replay"],
    ["plot"],
    ["db", "create"],
    ["db", "update"],
    ["db", "check"],
    ["optimize", "transfers"],
    ["optimize", "squad"],
    ["apply", "transfers"],
    ["apply", "lineup"],
    ["env", "get"],
    ["env", "set"],
    ["env", "names"],
    ["dump", "transfermarkt"],
]


class TestFlagSpelling:
    """
    Every boolean option reads as the thing it turns on.

    `airsenal predict --help` used to offer `--no-bonus --no-no-bonus
    [default: no-no-bonus]`, five times over, because the parameters were named
    for the negative. Typer derives `--no-X` itself, so the parameter has to be
    the positive.
    """

    @pytest.mark.parametrize("command", ALL_COMMANDS, ids=" ".join)
    def test_no_option_is_a_double_negative(self, command):
        result = runner.invoke(app, [*command, "--help"])
        assert result.exit_code == 0
        assert "no-no-" not in _flatten(result.stdout)

    @pytest.mark.parametrize(
        ("command", "positive"),
        [
            (["predict"], "--bonus"),
            (["predict"], "--cards"),
            (["predict"], "--saves"),
            (["predict"], "--def-con"),
            (["run"], "--current-season"),
            (["run"], "--subs"),
            (["optimize", "squad"], "--subs"),
            (["optimize", "transfers"], "--subs"),
        ],
    )
    def test_the_positive_and_its_negation_are_both_offered(self, command, positive):
        result = runner.invoke(app, [*command, "--help"])
        text = _flatten(result.stdout)
        assert positive in text
        assert f"--no-{positive.removeprefix('--')}" in text

    def test_apply_asks_before_acting_unless_told_not_to(self):
        """
        `--confirm` used to *skip* the confirmation, saying the opposite of what
        it did, and shadowing `core.console.confirm`.
        """
        text = _flatten(runner.invoke(app, ["apply", "transfers", "--help"]).stdout)
        assert "--yes" in text
        assert "--confirm" not in text

    def test_an_unattended_run_can_apply_without_a_prompt(self):
        """`run --apply-transfers` blocked on a prompt with no way to skip it."""
        text = _flatten(runner.invoke(app, ["run", "--help"]).stdout)
        assert "--apply-transfers" in text
        assert "--yes" in text

    def test_the_internal_persistence_mode_is_not_offered(self):
        """`--is-replay` is how `replay` stores suggestions, not a user choice."""
        for command in (["optimize", "transfers"], ["optimize", "squad"]):
            text = _flatten(runner.invoke(app, [*command, "--help"]).stdout)
            assert "--is-replay" not in text


class TestSettingsWithNoFlag:
    """Settings whose docstrings describe them as things people want."""

    @pytest.mark.parametrize(
        ("command", "option"),
        [
            # "re-optimise without re-fetching"
            (["run"], "--no-refresh-database"),
            # the mode that exists for an unattended run, previously unreachable
            (["run"], "--on-stale"),
            (["run"], "--gameweek-start"),
            # threaded through two layers but absent from run_prediction's
            # signature, so it could never be False
            (["predict"], "--no-def-con"),
            # the chip block was missing from replay entirely
            (["replay"], "--wildcard-week"),
        ],
    )
    def test_option_is_offered(self, command, option):
        result = runner.invoke(app, [*command, "--help"])
        assert result.exit_code == 0
        assert option in _flatten(result.stdout)


def test_env_names_prints_one_key_per_line():
    """It used to print a Python list repr."""
    result = runner.invoke(app, ["env", "names"])
    assert result.exit_code == 0
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert "FPL_TEAM_ID" in lines
    assert all("[" not in line for line in lines)


def _flatten(text: str) -> str:
    """Rich wraps help output, so join it back up before searching for a flag."""
    return " ".join(text.split())
