"""Regression tests for pandas compatibility in data-processing scripts."""

import json
import warnings

import pandas as pd

from airsenal.scripts import make_results
from airsenal.scripts.scrape_transfermarkt import tidy_df


def test_make_results_replaces_team_ids_without_chained_assignment(
    tmp_path, monkeypatch
):
    season = "2526"
    fixture_pattern = str(tmp_path / "fixture_data_{}.json")
    summary_pattern = str(tmp_path / "summary_data_{}.json")
    results_pattern = str(tmp_path / "results_{}.csv")

    (tmp_path / f"fixture_data_{season}.json").write_text(
        json.dumps(
            [
                {
                    "kickoff_time": "2025-08-16T14:00:00Z",
                    "team_h": 1,
                    "team_a": 2,
                    "team_h_score": 2,
                    "team_a_score": 1,
                    "event": 1,
                }
            ]
        )
    )
    (tmp_path / f"summary_data_{season}.json").write_text(
        json.dumps(
            {"teams": [{"id": 1, "name": "Arsenal"}, {"id": 2, "name": "Chelsea"}]}
        )
    )
    monkeypatch.setattr(make_results, "FIXTURE_DATA_FILE", fixture_pattern)
    monkeypatch.setattr(make_results, "SUMMARY_DATA_FILE", summary_pattern)
    monkeypatch.setattr(make_results, "RESULTS_FILE", results_pattern)

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        make_results.make_results(season)

    result = pd.read_csv(results_pattern.format(season))
    assert result.loc[0, "home_team"] == "Arsenal"
    assert result.loc[0, "away_team"] == "Chelsea"


def test_tidy_df_replaces_missing_markers_without_downcast_warning():
    raw = pd.DataFrame(
        {
            "Season": ["24/25", "24/25"],
            "From": ["01/01/2025", "02/01/2025"],
            "Until": ["03/01/2025", "04/01/2025"],
            "Days": ["2 days", "? days"],
            "Games missed": [1, "-"],
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        result = tidy_df(raw)

    assert result["games"].iloc[0] == 1
    assert pd.isna(result["games"].iloc[1])
    assert result["days"].iloc[0] == 2
    assert pd.isna(result["days"].iloc[1])
