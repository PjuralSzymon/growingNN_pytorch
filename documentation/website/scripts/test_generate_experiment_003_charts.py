"""Tests for the Experiment 003 chart generator."""

import json
from pathlib import Path

from generate_experiment_003_charts import (
    DEFAULT_AFTER_RUNS,
    DEFAULT_BEFORE_RUNS,
    DEFAULT_RUNS,
    DEFAULT_SNAPSHOT,
    PHASE_AFTER,
    PHASE_BEFORE,
    generate_charts,
    load_runs,
)


def _write_board(
    root: Path,
    score_metric: str,
    model: str,
    seed: int,
    dropout: bool,
) -> None:
    board = root / score_metric / model / "config" / f"seed_{seed}" / "board"
    (board / "metrics").mkdir(parents=True)
    epochs = [
        {
            "globalEpoch": generation * 10 + epoch,
            "generation": generation,
            "trainAcc": 0.2 + generation * 0.1,
            "valAcc": 0.25 + generation * 0.1,
            "paramCount": 400 + generation * 50,
        }
        for generation in range(5)
        for epoch in range(10)
    ]
    (board / "main.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "trainingTimeElapsedSec": 10,
                "generationTimeline": [
                    {
                        "generation": 0,
                        "actionExecuted": {
                            "shortLabel": (
                                "Add Seq Dropout Layer Action"
                                if dropout
                                else "Add Res Conv Layer Action"
                            ),
                            "atGlobalEpoch": 9,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (board / "metrics" / "training.json").write_text(
        json.dumps({"epochs": epochs}),
        encoding="utf-8",
    )


def test_default_runs_point_at_before_and_after_fix_folders() -> None:
    """
    Exp 003 charts should read the historical before folder and after_fix_1 folder.
    """
    # Arrange / Act / Assert
    assert DEFAULT_BEFORE_RUNS.name == "exp003_score_accuracy_metric"
    assert DEFAULT_AFTER_RUNS.name == "exp003_score_accuracy_metric_after_fix_1"
    assert DEFAULT_RUNS == DEFAULT_BEFORE_RUNS


def test_load_runs_reads_score_metric_model_seed_board(tmp_path: Path) -> None:
    """
    The loader should derive score_metric, model, and seed from the board path.
    """
    # Arrange
    board = tmp_path / "train_acc" / "big" / "config" / "seed_100" / "board"
    (board / "metrics").mkdir(parents=True)
    (board / "main.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "trainingTimeElapsedSec": 12,
                "generationTimeline": [
                    {
                        "generation": 0,
                        "actionExecuted": {
                            "shortLabel": "Add Seq Dropout Layer Action",
                            "atGlobalEpoch": 9,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (board / "metrics" / "training.json").write_text(
        json.dumps(
            {
                "epochs": [
                    {
                        "globalEpoch": 0,
                        "generation": 0,
                        "trainAcc": 0.2,
                        "valAcc": 0.25,
                        "paramCount": 420,
                    },
                    {
                        "globalEpoch": 1,
                        "generation": 0,
                        "trainAcc": 0.5,
                        "valAcc": 0.55,
                        "paramCount": 420,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    # Act
    runs = load_runs(tmp_path)

    # Assert
    assert len(runs) == 1
    assert runs[0]["score_metric"] == "train_acc"
    assert runs[0]["model"] == "big"
    assert runs[0]["seed"] == 100
    assert runs[0]["dropout_actions"] == 1
    assert runs[0]["final_acc"] == 0.55


def test_generate_charts_writes_before_after_compare_and_snapshot(tmp_path: Path) -> None:
    """
    generate_charts should write before/after/compare PNGs and a phased snapshot.
    """
    # Arrange
    before_dir = tmp_path / "before"
    after_dir = tmp_path / "after"
    output_dir = tmp_path / "out"
    snapshot = tmp_path / "snapshot.json"
    for score_metric, model, seed in (
        ("val_acc", "big", 100),
        ("val_acc", "medium_1conv_2linear", 100),
        ("train_acc", "big", 100),
        ("train_acc", "medium_1conv_2linear", 101),
    ):
        _write_board(before_dir, score_metric, model, seed, dropout=True)
        _write_board(after_dir, score_metric, model, seed, dropout=False)

    # Act
    written = generate_charts(
        before_runs_dir=before_dir,
        after_runs_dir=after_dir,
        output_dir=output_dir,
        snapshot_path=snapshot,
    )

    # Assert
    names = {path.name for path in written}
    assert "003-before-final-accuracy-by-score-metric.png" in names
    assert "003-after-final-accuracy-by-score-metric.png" in names
    assert "003-before-grading-overall-final-validation.png" in names
    assert "003-after-grading-overall-final-validation.png" in names
    assert "003-before-grading-by-model-final-validation.png" in names
    assert "003-after-grading-by-model-final-validation.png" in names
    assert "003-before-dropout-actions-by-score-metric.png" in names
    assert "003-after-dropout-actions-by-score-metric.png" in names
    assert "003-before-action-composition-by-score-metric.png" in names
    assert "003-before-action-types.png" in names
    assert "003-before-action-types-by-score-metric.png" in names
    assert "003-before-training-curves.png" in names
    assert "003-compare-final-validation-by-score-metric.png" in names
    assert "003-compare-dropout-actions-by-score-metric.png" in names
    assert "003-compare-overall-before-after.png" in names
    assert "003-compare-grading-overall-by-phase.png" in names
    assert "003-final-accuracy-by-score-metric.png" in names
    assert snapshot.exists()
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    assert len(payload[PHASE_BEFORE]["runs"]) == 4
    assert len(payload[PHASE_AFTER]["runs"]) == 4
    assert len(payload["runs"]) == 4
    assert DEFAULT_SNAPSHOT.name.endswith(".json")


def test_generate_charts_before_only_skips_compare(tmp_path: Path) -> None:
    """
    When only before_fix exists, phase charts write and compare charts are skipped.
    """
    # Arrange
    before_dir = tmp_path / "before"
    after_dir = tmp_path / "after_missing"
    output_dir = tmp_path / "out"
    snapshot = tmp_path / "snapshot.json"
    _write_board(before_dir, "val_acc", "big", 100, dropout=True)

    # Act
    written = generate_charts(
        before_runs_dir=before_dir,
        after_runs_dir=after_dir,
        output_dir=output_dir,
        snapshot_path=snapshot,
    )

    # Assert
    names = {path.name for path in written}
    assert "003-before-final-accuracy-by-score-metric.png" in names
    assert "003-compare-final-validation-by-score-metric.png" not in names
    assert "003-after-final-accuracy-by-score-metric.png" not in names
