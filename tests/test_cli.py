import pytest
from typer.testing import CliRunner

from airsenal.cli.main import app
from airsenal.core.registry import ConfigError
from airsenal.prediction.models import (
    PLAYER_MODELS,
    TEAM_MODELS,
    build_player_model,
    build_team_model,
)

runner = CliRunner()


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


def _flatten(text: str) -> str:
    """Rich wraps help output, so join it back up before searching for a flag."""
    return " ".join(text.split())
