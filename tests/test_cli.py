import pytest
from typer.testing import CliRunner

from airsenal.cli.main import app
from airsenal.core.registry import ConfigError
from airsenal.optimization.config import GeneticAlgorithmConfig
from airsenal.optimization.run_squad import build_ga_config
from airsenal.prediction.registry import PLAYER_MODELS, TEAM_MODELS

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
    The registries reach the user through these options.

    Before them, the only model choice was a `--sampling` boolean, and the
    hyperparameters it implied were silently dropped.
    """

    @pytest.mark.parametrize(
        ("command", "option"),
        [
            (["predict"], "--player-model"),
            (["predict"], "--team-model"),
            (["predict"], "--set-player"),
            (["predict"], "--set-team"),
            (["run"], "--player-model"),
            (["run"], "--set-player"),
            (["run"], "--set-team"),
            (["replay"], "--player-model"),
            (["replay"], "--set-player"),
            (["replay"], "--set-team"),
            (["optimize", "squad"], "--set-ga"),
            (["optimize", "transfers"], "--save-strategies"),
            (["optimize", "transfers"], "--set-ga"),
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
        for name in (*PLAYER_MODELS.names(), *TEAM_MODELS.names()):
            assert name in text

    def test_unknown_player_model_lists_the_available_ones(self):
        with pytest.raises(ConfigError, match="Choose from: conjugate, constant"):
            PLAYER_MODELS.create_with("nope", {})

    def test_unknown_option_lists_the_available_ones(self):
        with pytest.raises(ConfigError, match="no option\\(s\\) nope"):
            PLAYER_MODELS.create_with("conjugate", {"nope": "1"})

    def test_ga_defaults_come_from_the_config_not_the_cli(self):
        # The CLI used to restate all nine GA defaults, so they could drift from
        # GeneticAlgorithmConfig without anything noticing.
        default = GeneticAlgorithmConfig()
        assert build_ga_config(None, None, None) == default
        assert build_ga_config(7, None, None).generations == 7
        assert build_ga_config(None, 9, None).population_size == 9
        tuned = build_ga_config(None, None, {"tournament_size": "5"})
        assert tuned.tournament_size == 5

    def test_first_class_flags_win_over_set_ga(self):
        config = build_ga_config(7, 9, {"generations": "3", "population_size": "4"})
        assert (config.generations, config.population_size) == (7, 9)


def _flatten(text: str) -> str:
    """Rich wraps help output, so join it back up before searching for a flag."""
    return " ".join(text.split())
