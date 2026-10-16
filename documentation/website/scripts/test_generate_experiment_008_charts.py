"""Unit tests for Experiment 008 chart helpers."""

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from documentation.website.scripts.generate_experiment_008_charts import (
    experiment_has_finished,
    load_measurement,
    plot_history,
    plot_parameters,
    plot_phase2_group_means,
    plot_phase2_mode_grid,
    plot_phase2_official_test,
    plot_phase2_param_grid,
    plot_phase3_mode_grid,
    plot_phase3_parameters,
    write_charts,
)


def _history(test: float) -> dict[str, object]:
    return {
        "train_acc": [0.2, 0.4],
        "val_acc": [0.25, 0.35],
        "lr": [0.1, 0.001],
        "param_count": [10, 20],
        "test_acc": [0.3, test],
        "test_epoch": [10, 20],
        "action_generation": [0],
        "action_name": ["AddSeqConvLayer"],
        "action_val_acc_before": [0.25],
        "action_val_acc_after": [0.25],
    }


def _write_fake_run(run_dir: Path) -> None:
    for seed, test in ((110, 0.91), (111, 0.90), (112, 0.92)):
        seed_dir = run_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True)
        torch.save(_history(test), seed_dir / "history.pt")


def test_experiment_has_finished_requires_every_seed(tmp_path: Path):
    """
    experiment_has_finished should stay false until all three seeds have history.pt.
    """

    # Arrange
    missing_dir = tmp_path / "partial"
    (missing_dir / "seed_110").mkdir(parents=True)
    (missing_dir / "seed_110" / "history.pt").write_bytes(b"x")
    ready_dir = tmp_path / "ready"
    _write_fake_run(ready_dir)

    # Act
    missing = experiment_has_finished(missing_dir)
    ready = experiment_has_finished(ready_dir)

    # Assert
    assert missing is False
    assert ready is True


def test_load_measurement_reads_generation_end_test(tmp_path: Path):
    """
    load_measurement should read official test at the saved epochs and the executed action.
    """

    # Arrange
    run_dir = tmp_path / "runs"
    _write_fake_run(run_dir)

    # Act
    data = load_measurement(run_dir, tmp_path / "missing.json")

    # Assert
    assert data["runs"][0]["final_test"] == 0.91
    assert data["runs"][0]["test_epoch"] == [10, 20]
    assert data["runs"][0]["best_generation"] == 1
    assert data["runs"][0]["actions"][0]["name"] == "AddSeqConvLayer"
    assert data["runs"][2]["params_end"] == 20


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


def test_plot_parameters_writes_the_parameter_chart(tmp_path: Path):
    """
    plot_parameters should write the parameter-count chart when every seed has param_count.
    """

    # Arrange
    data = {
        "runs": [
            {"seed": seed, "param_count": [10, 20]}
            for seed in (110, 111, 112)
        ],
    }

    # Act
    plot_parameters(data, tmp_path)

    # Assert
    assert (tmp_path / "008-parameters.png").is_file()


def test_plot_history_writes_one_row_for_train_test_and_validation(tmp_path: Path):
    """
    plot_history should write phase 1 train, official test, and validation as one row.
    """

    # Arrange
    data = {
        "runs": [
            {
                "seed": seed,
                "train": [0.3, 0.8],
                "val": [0.4, 0.7],
                "test": [0.5, 0.75],
                "test_epoch": [10, 20],
            }
            for seed in (110, 111, 112)
        ]
    }

    # Act
    plot_history(data, tmp_path)

    # Assert
    assert (tmp_path / "008-history.png").is_file()


def test_plot_phase2_group_means_writes_the_four_group_chart(tmp_path: Path):
    """
    plot_phase2_group_means should write one chart of mean final accuracy for all four groups.
    """

    # Arrange
    def rows(test: float) -> list[dict[str, object]]:
        return [
            {"final_train": 0.9, "final_val": 0.8, "final_test": test, "train": [0.9], "val": [0.8], "test": [test]}
            for _seed in (110, 111, 112)
        ]

    data = {
        "runs": rows(0.91),
        "phase2_all": rows(0.85),
        "phase2_grow": rows(0.86),
        "phase2_seq_res": rows(0.83),
    }

    # Act
    plot_phase2_group_means(data, tmp_path)

    # Assert
    assert (tmp_path / "008-phase2-group-means.png").is_file()


def test_plot_phase2_mode_grid_writes_four_by_three_accuracy_panels(tmp_path: Path):
    """
    plot_phase2_mode_grid should write one panel per action group and accuracy.
    """

    # Arrange
    rows = [
        {
            "seed": seed,
            "train": [0.3, 0.8],
            "val": [0.4, 0.7],
            "test": [0.5, 0.75],
            "test_epoch": [10, 20],
        }
        for seed in (110, 111, 112)
    ]
    data = {
        "runs": rows,
        "phase2_all": rows,
        "phase2_grow": rows,
        "phase2_seq_res": rows,
    }

    # Act
    plot_phase2_mode_grid(data, tmp_path)

    # Assert
    assert (tmp_path / "008-phase2-mode-grid.png").is_file()


def test_plot_phase2_param_grid_writes_one_panel_per_action_group(tmp_path: Path):
    """
    plot_phase2_param_grid should write one parameter-count panel for each of the four groups.
    """

    # Arrange
    rows = [
        {"seed": seed, "param_count": [187008, 187008 + seed]}
        for seed in (110, 111, 112)
    ]
    data = {
        "runs": rows,
        "phase2_all": rows,
        "phase2_grow": rows,
        "phase2_seq_res": rows,
    }

    # Act
    plot_phase2_param_grid(data, tmp_path)

    # Assert
    assert (tmp_path / "008-phase2-param-grid.png").is_file()


def test_plot_phase2_official_test_writes_the_comparison_chart(tmp_path: Path):
    """
    plot_phase2_official_test should write the phase 1 versus all-group chart when phase2_all is present.
    """

    # Arrange
    data = {
        "runs": [
            {"seed": seed, "test": [0.5, 0.9], "test_epoch": [10, 20]}
            for seed in (110, 111, 112)
        ],
        "phase2_all": [
            {"seed": seed, "test": [0.4, 0.8], "test_epoch": [10, 20]}
            for seed in (110, 111, 112)
        ],
    }

    # Act
    plot_phase2_official_test(data, tmp_path)

    # Assert
    assert (tmp_path / "008-phase2-official-test.png").is_file()


def test_plot_phase3_mode_grid_writes_three_budgets_by_three_accuracies(tmp_path: Path):
    """
    plot_phase3_mode_grid should write train, official test, and validation for each saved budget.
    """

    # Arrange
    rows = [
        {
            "seed": seed,
            "train": [0.3, 0.8],
            "val": [0.4, 0.7],
            "test": [0.5, 0.9],
            "test_epoch": [10, 20],
        }
        for seed in (110, 111, 112)
    ]
    data = {"runs": rows, "phase2_all": rows, "phase3_all": rows}

    # Act
    plot_phase3_mode_grid(data, tmp_path)

    # Assert
    assert (tmp_path / "008-phase3-mode-grid.png").is_file()


def test_plot_phase3_parameters_writes_the_long_all_count(tmp_path: Path):
    """
    plot_phase3_parameters should write the long all-group parameter count when those rows exist.
    """

    # Arrange
    data = {
        "phase3_all": [
            {"seed": seed, "param_count": [187008, 200000 + seed]}
            for seed in (110, 111, 112)
        ]
    }

    # Act
    plot_phase3_parameters(data, tmp_path)

    # Assert
    assert (tmp_path / "008-phase3-parameters.png").is_file()


def test_write_charts_falls_back_to_snapshot(tmp_path: Path):
    """
    write_charts should draw from the snapshot when the raw run directory is absent.
    """

    # Arrange
    snapshot = tmp_path / "snapshot.json"
    payload = {
        "experiment": "008",
        "seeds": [110, 111, 112],
        "runs": [
            {
                "seed": seed,
                "train": [0.2, 0.4],
                "val": [0.25, 0.35],
                "lr": [0.1, 0.001],
                "test": [0.5, 0.9],
                "test_epoch": [10, 20],
            }
            for seed in (110, 111, 112)
        ],
    }
    original = json.dumps(payload)
    snapshot.write_text(original, encoding="utf-8")
    out_dir = tmp_path / "charts"

    # Act
    write_charts(out_dir, snapshot, tmp_path / "missing-run")

    # Assert
    assert (out_dir / "008-official-test.png").is_file()
    assert (out_dir / "008-history.png").is_file()
    assert (out_dir / "008-learning-rate.png").is_file()
    assert snapshot.read_text(encoding="utf-8") == original
