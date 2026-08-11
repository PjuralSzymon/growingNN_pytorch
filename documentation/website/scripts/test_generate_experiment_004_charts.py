"""Unit tests for Experiment 004 chart helpers."""

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from documentation.website.scripts.generate_experiment_004_charts import (
    SCHEDULE_IDS,
    autoscaled_y_limits,
    generate_charts,
    simulate_lr_components,
)


def test_autoscaled_y_limits_fit_small_learning_rate_range():
    """
    autoscaled_y_limits should zoom into a 0.001..0.01 band instead of forcing 0..1.
    """
    # Arrange
    values = [0.001, 0.005, 0.01, 0.008]

    # Act
    low, high = autoscaled_y_limits(values)

    # Assert
    assert low < 0.001
    assert high > 0.01
    assert high < 0.5


def test_simulate_lr_components_drops_recovery_after_action():
    """
    simulate_lr_components should lower the recovery factor right after an action reset epoch.
    """
    # Arrange / Act
    components = simulate_lr_components("composed_constant", action_generations=[1], n_epochs=40)

    # Assert
    assert components["factor"][19] == pytest.approx(1.0, abs=0.05)
    assert components["factor"][20] < 0.2
    assert components["effective"][20] < components["base"][20]


def test_generate_charts_writes_dual_seed_metric_panels(tmp_path: Path):
    """
    generate_charts should write LR/train/val panels for seeds 100 and 101 per schedule.
    """
    # Arrange
    output_dir = tmp_path / "out"
    snapshot = tmp_path / "snap.json"
    synthetic_runs = []
    for schedule_id in SCHEDULE_IDS:
        for seed in (100, 101, 102):
            synthetic_runs.append(
                {
                    "schedule_id": schedule_id,
                    "seed": seed,
                    "status": "completed",
                    "actions": 1,
                    "action_generations": [1],
                    "action_labels": ["Add Res Conv Layer Action"],
                    "train_acc": [0.2 + 0.001 * i for i in range(40)],
                    "val_acc": [0.25 + 0.001 * i for i in range(40)],
                    "lr": [0.01 - 0.0001 * i for i in range(40)],
                    "final_train_acc": 0.4,
                    "final_val_acc": 0.45,
                }
            )
    snapshot.write_text(
        json.dumps({"experiment": "004", "folder": "tmp", "runs": synthetic_runs}),
        encoding="utf-8",
    )

    # Act
    written = generate_charts(
        runs_dir=tmp_path / "missing_raw",
        output_dir=output_dir,
        snapshot_path=snapshot,
    )
    names = {path.name for path in written}

    # Assert
    assert "004-scheduler-shape-guide.png" in names
    assert "004-lr-composed_cosine-seeds-100-101.png" in names
    assert "004-train-acc-composed_step-seeds-100-101.png" in names
    assert "004-val-acc-recovery_only_logistic-seeds-100-101.png" in names
    assert "004-post-action-train-acc-change-by-schedule.png" in names
    assert "004-final-accuracy-by-schedule.png" in names


def test_generate_charts_still_writes_shape_guide_without_runs(tmp_path: Path):
    """
    generate_charts should still write the schedule shape guide when no boards exist.
    """
    # Arrange / Act
    written = generate_charts(
        runs_dir=tmp_path / "missing",
        output_dir=tmp_path / "out",
        snapshot_path=tmp_path / "missing.json",
    )
    names = {path.name for path in written}

    # Assert
    assert names == {"004-scheduler-shape-guide.png"}


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
