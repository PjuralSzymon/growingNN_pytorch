"""Tests for the Experiment 002 chart generator."""

import json
from pathlib import Path

from generate_experiment_002_charts import DEFAULT_SNAPSHOT, generate_charts, load_runs


def test_load_runs_reads_architecture_board_result(tmp_path: Path) -> None:
    """
    The loader should derive one normalized row from model/seed board JSON.
    """
    # Arrange
    board = tmp_path / "medium" / "config" / "seed_100" / "board"
    (board / "metrics").mkdir(parents=True)
    (board / "main.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "trainingTimeElapsedSec": 12,
                "experimentStartedOn": "2026-08-04T12:00:00Z",
                "lastUpdate": "2026-08-04T13:00:00Z",
                "generationTimeline": [
                    {
                        "generation": 1,
                        "actionExecuted": {
                            "shortLabel": "Add Seq Conv Layer Action",
                            "atGlobalEpoch": 20,
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
                        "lr": 0.01,
                        "paramCount": 276,
                    },
                    {
                        "globalEpoch": 1,
                        "generation": 0,
                        "trainAcc": 0.7,
                        "valAcc": 0.65,
                        "lr": 0.01,
                        "paramCount": 500,
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
    assert runs[0]["model"] == "medium"
    assert runs[0]["seed"] == 100
    assert runs[0]["actions"] == 1
    assert runs[0]["action_labels"] == ["Add Seq Conv Layer Action"]
    assert runs[0]["action_generations"] == [1]
    assert runs[0]["final_acc"] == 0.65
    assert runs[0]["best_acc"] == 0.65
    assert runs[0]["start_params"] == 276


def test_generate_charts_writes_core_figures_and_snapshot(tmp_path: Path) -> None:
    """
    generate_charts should write the kept Experiment 002 PNGs and refresh the snapshot.
    """
    # Arrange
    runs_dir = tmp_path / "runs"
    output_dir = tmp_path / "out"
    snapshot = tmp_path / "snapshot.json"
    for model, seed, start_params in (
        ("very_small", 100, 76),
        ("medium_h4", 100, 96),
        ("medium", 100, 276),
        ("big", 100, 420),
        ("big", 101, 420),
    ):
        board = runs_dir / model / "config" / f"seed_{seed}" / "board"
        (board / "metrics").mkdir(parents=True)
        epochs = [
            {
                "globalEpoch": generation * 10 + epoch,
                "generation": generation,
                "trainAcc": 0.2 + generation * 0.05 + epoch * 0.01,
                "valAcc": 0.2 + generation * 0.05 + epoch * 0.01,
                "lr": 0.001 if generation in (1, 3) and epoch == 0 else 0.01,
                "paramCount": start_params + generation * 100,
            }
            for generation in range(10)
            for epoch in range(10)
        ]
        (board / "main.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "trainingTimeElapsedSec": 10,
                    "generationTimeline": [
                        {
                            "generation": generation,
                            "actionExecuted": (
                                {
                                    "shortLabel": (
                                        "Add Seq Conv Layer Action"
                                        if generation == 0 and model == "very_small"
                                        else "Add Res Conv Layer Action"
                                    ),
                                    "atGlobalEpoch": generation * 10 + 9,
                                }
                                if generation in (0, 1, 3, 5, 8)
                                else None
                            ),
                        }
                        for generation in range(10)
                    ],
                }
            ),
            encoding="utf-8",
        )
        (board / "metrics" / "training.json").write_text(
            json.dumps({"epochs": epochs}),
            encoding="utf-8",
        )

    # Act
    written = generate_charts(
        runs_dir=runs_dir,
        output_dir=output_dir,
        snapshot_path=snapshot,
    )

    # Assert
    names = {path.name for path in written}
    assert "002-final-accuracy-by-architecture.png" in names
    assert "002-actions-by-phase.png" in names
    assert "002-action-order.png" in names
    assert "002-action-types.png" in names
    assert "002-param-growth.png" in names
    assert "002-peak-vs-final.png" in names
    assert "002-training-curves.png" in names
    assert "002-representative-timeline.png" not in names
    assert "002-slope-decisions.png" not in names
    assert snapshot.exists()
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    assert len(payload["runs"]) == 5
    assert DEFAULT_SNAPSHOT.name.endswith(".json")


def test_generate_charts_falls_back_to_snapshot_when_raw_missing(tmp_path: Path) -> None:
    """
    When raw board output is absent, generate_charts should read the snapshot JSON.
    """
    # Arrange
    snapshot = tmp_path / "snapshot.json"
    output_dir = tmp_path / "out"
    snapshot.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "model": "big",
                        "seed": 100,
                        "status": "completed",
                        "elapsed_sec": 1,
                        "final_acc": 0.9,
                        "final_train_acc": 0.88,
                        "best_acc": 0.91,
                        "best_train_acc": 0.89,
                        "final_params": 900,
                        "start_params": 420,
                        "actions": 1,
                        "action_generations": [1],
                        "action_epochs": [19],
                        "action_labels": ["Add Res Conv Layer Action"],
                        "epochs": [
                            {
                                "globalEpoch": generation * 10 + epoch,
                                "generation": generation,
                                "trainAcc": 0.3 + generation * 0.05,
                                "valAcc": 0.3 + generation * 0.05,
                                "lr": 0.01,
                                "paramCount": 420 + generation * 50,
                            }
                            for generation in range(10)
                            for epoch in range(10)
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    # Act
    written = generate_charts(
        runs_dir=tmp_path / "missing_raw",
        output_dir=output_dir,
        snapshot_path=snapshot,
    )

    # Assert
    assert written
    assert (output_dir / "002-final-accuracy-by-architecture.png").exists()
    assert (output_dir / "002-actions-by-phase.png").exists()
