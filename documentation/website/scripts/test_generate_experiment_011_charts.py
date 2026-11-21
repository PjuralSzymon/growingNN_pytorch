"""Tests for the Experiment 011 chart generator."""

import json
from pathlib import Path

from generate_experiment_011_charts import generate_charts, load_runs


def _snapshot(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "experiment": "011_simulation_sets",
                "timeline_source": "filesystem mtime of history.pt",
                "runs": [
                    {
                        "algorithm": "moderate",
                        "seed": 1,
                        "written": "2026-11-20 14:08:35",
                        "train": [0.2, 0.4],
                        "val": [0.3, 0.5],
                        "test": [0.31],
                        "params_generation": [500],
                        "final_train": 0.4,
                        "final_val": 0.5,
                        "final_test": 0.31,
                        "best_val": 0.5,
                        "params_end": 500,
                        "actions": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_load_runs_uses_the_snapshot_when_raw_histories_are_absent(tmp_path: Path):
    """
    load_runs should read the JSON snapshot when the run folder has no history.pt files.
    """

    # Arrange
    snapshot = tmp_path / "experiment-011-simulation-sets.json"
    _snapshot(snapshot)

    # Act
    data = load_runs(tmp_path / "missing-runs", snapshot)

    # Assert
    assert data["runs"][0]["algorithm"] == "moderate"
    assert data["runs"][0]["final_val"] == 0.5


def test_generate_charts_writes_both_figures_from_the_snapshot(tmp_path: Path):
    """
    generate_charts should write the accuracy, history, and mean-curve figures from snapshot rows.
    """

    # Arrange
    snapshot = tmp_path / "experiment-011-simulation-sets.json"
    out_dir = tmp_path / "assets"
    _snapshot(snapshot)

    # Act
    generate_charts(tmp_path / "missing-runs", snapshot, out_dir)

    # Assert
    assert (out_dir / "011-final-accuracy.svg").is_file()
    assert (out_dir / "011-final-accuracy-cifar10.svg").is_file()
    assert (out_dir / "011-training-histories.svg").is_file()
    assert (out_dir / "011-mean-training-accuracy.svg").is_file()
    assert (out_dir / "011-mean-validation-accuracy.svg").is_file()
