import json

import pandas as pd

from airsenal.export import results


def test_the_results_file_names_both_teams(monkeypatch, tmp_path):
    """Home and away team ids are written out as the teams' names."""
    monkeypatch.setattr(
        results, "FIXTURE_DATA_FILE", str(tmp_path / "fixture_data_{}.json")
    )
    monkeypatch.setattr(results, "SUMMARY_DATA_FILE", str(tmp_path / "FPL_{}.json"))
    monkeypatch.setattr(results, "RESULTS_FILE", str(tmp_path / "results_{}.csv"))
    fixture = {
        "kickoff_time": "2025-08-15T19:00:00Z",
        "team_h": 1,
        "team_a": 2,
        "team_h_score": 4,
        "team_a_score": 2,
        "event": 1,
    }
    (tmp_path / "fixture_data_2526.json").write_text(json.dumps([fixture]))
    teams = [{"id": 1, "name": "Liverpool"}, {"id": 2, "name": "Bournemouth"}]
    (tmp_path / "FPL_2526.json").write_text(json.dumps({"teams": teams}))

    results.make_results("2526")

    written = pd.read_csv(tmp_path / "results_2526.csv")
    assert written["home_team"].tolist() == ["Liverpool"]
    assert written["away_team"].tolist() == ["Bournemouth"]
