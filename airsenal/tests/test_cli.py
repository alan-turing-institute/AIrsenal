import os

from typer.testing import CliRunner

from airsenal.cli.main import app

# Typer forces rich's terminal/color output on when GITHUB_ACTIONS is set (so CLI
# help looks nice in workflow logs), which injects ANSI escape codes into
# result.stdout and breaks the plain substring assertions below. Disable that
# forcing so these tests behave the same locally and on GitHub Actions.
os.environ["_TYPER_FORCE_DISABLE_TERMINAL"] = "1"


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
