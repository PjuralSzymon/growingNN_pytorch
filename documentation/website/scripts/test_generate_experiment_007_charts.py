"""Unit tests for Experiment 007 chart helpers."""

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from documentation.website.scripts.generate_experiment_007_charts import (
    _load_added_epoch_history,
    experiment_has_finished,
    load_measurement,
    plot_final_bars,
    write_charts,
)


def _history() -> dict[str, list[object]]:
    return {
        "train_acc": [0.2, 0.3],
        "val_acc": [0.25, 0.35],
        "test_acc": [0.4],
        "param_count": [10, 10],
        "action_generation": [1],
        "action_name": ["AddResConvLayer"],
        "action_val_acc_before": [0.3],
        "action_val_acc_after": [0.31],
    }


def _write_fake_run(run_dir: Path) -> None:
    for seed in (110, 111, 112):
        original = run_dir / "original" / "original" / f"seed_{seed}"
        original.mkdir(parents=True)
        torch.save(_history(), original / "history.pt")
        added = run_dir / "added" / f"seed_{seed}"
        added.mkdir(parents=True)
        state = {"seq_conv_0.weight": torch.zeros(1), "seq_conv_0_bn.weight": torch.zeros(1)}
        torch.save({"state_dict": state, "test_acc": [0.5, 0.6]}, added / "added_growth.pt")
        torch.save({"state_dict": state, "test_acc": 0.91, "chunk": 2, "seed": seed}, added / "added_finish_best.pt")
        torch.save({"state_dict": state, "test_acc": 0.9, "seed": seed}, added / "added_finish.pt")


def test_experiment_has_finished_requires_every_seed(tmp_path: Path):
    """
    experiment_has_finished should stay false until all three seeds have both result files.
    """

    # Arrange
    missing_dir = tmp_path / "partial"
    (missing_dir / "original" / "original" / "seed_110").mkdir(parents=True)
    (missing_dir / "original" / "original" / "seed_110" / "history.pt").write_bytes(b"x")
    ready_dir = tmp_path / "ready"
    _write_fake_run(ready_dir)

    # Act
    missing = experiment_has_finished(missing_dir)
    ready = experiment_has_finished(ready_dir)

    # Assert
    assert missing is False
    assert ready is True


def test_load_measurement_reads_histories_and_finish_checkpoints(tmp_path: Path):
    """
    load_measurement should read original actions and the added best finish chunk.
    """

    # Arrange
    run_dir = tmp_path / "runs"
    _write_fake_run(run_dir)

    # Act
    data = load_measurement(run_dir, tmp_path / "missing.json")

    # Assert
    assert data["original"][0]["final_test"] == 0.4
    assert data["original"][0]["actions"][0]["name"] == "AddResConvLayer"
    assert data["added"][0]["best_chunk"] == 2
    assert data["added"][0]["best_epoch"] == 56 + 15
    assert data["added"][0]["seq_conv_count"] == 1
    assert data["added"][0]["final_epoch"] == 56 + 120


def test_write_charts_refuses_before_the_experiment(tmp_path: Path):
    """
    write_charts should write no files when the run and the snapshot are both missing.
    """

    # Arrange
    out_dir = tmp_path / "charts"
    snapshot = tmp_path / "snapshot.json"

    # Act
    try:
        write_charts(out_dir, snapshot, tmp_path / "runs")
        refused = False
    except FileNotFoundError as exc:
        refused = "has not finished" in str(exc)

    # Assert
    assert refused is True
    assert not out_dir.exists()
    assert not snapshot.exists()


def test_plot_final_bars_writes_png(tmp_path: Path):
    """
    plot_final_bars should write the official-test bar chart from a small snapshot.
    """

    # Arrange
    data = {
        "original": [{"seed": 110, "final_test": 0.7}, {"seed": 111, "final_test": 0.6}, {"seed": 112, "final_test": 0.65}],
        "added": [
            {"best_test": 0.9, "final_test": 0.89},
            {"best_test": 0.88, "final_test": 0.87},
            {"best_test": 0.89, "final_test": 0.88},
        ],
    }

    # Act
    plot_final_bars(data, tmp_path)

    # Assert
    assert (tmp_path / "007-final-official-test.png").is_file()


def test_load_added_epoch_history_reads_train_val_and_test(tmp_path: Path):
    """
    Added history.pt should provide per-epoch train and validation and the saved test epochs.
    """

    # Arrange
    path = tmp_path / "history.pt"
    torch.save(
        {
            "train_acc": [0.2, 0.3],
            "val_acc": [0.15, 0.25],
            "test_acc": [0.4, 0.5],
            "test_epoch": [8, 16],
        },
        path,
    )

    # Act
    row = _load_added_epoch_history(path, 110)

    # Assert
    assert row["seed"] == 110
    assert row["train"] == [0.2, 0.3]
    assert row["val"] == [0.15, 0.25]
    assert row["test"] == [0.4, 0.5]
    assert row["test_epoch"] == [8, 16]


def test_write_charts_falls_back_to_snapshot(tmp_path: Path):
    """
    write_charts should draw from the snapshot when the raw run directory is absent.
    """

    # Arrange
    snapshot = tmp_path / "snapshot.json"
    payload = {
        "original": [
            {
                "seed": seed,
                "train": [0.2, 0.3],
                "val": [0.25, 0.35],
                "test": [0.4, 0.5],
                "test_epoch": [10, 20],
                "final_test": 0.5,
            }
            for seed in (110, 111, 112)
        ],
            "added": [
                {
                    "seed": seed,
                    "growth_test": [0.6, 0.7],
                    "growth_test_epoch": [8, 16],
                    "best_test": 0.9,
                    "final_test": 0.89,
                }
                for seed in (110, 111, 112)
            ],
    }
    snapshot.write_text(json.dumps(payload), encoding="utf-8")
    out_dir = tmp_path / "charts"

    # Act
    write_charts(out_dir, snapshot, tmp_path / "missing-run")

    # Assert
    assert (out_dir / "007-history-grid.png").is_file()
    assert not (out_dir / "007-official-test-curves.png").exists()
