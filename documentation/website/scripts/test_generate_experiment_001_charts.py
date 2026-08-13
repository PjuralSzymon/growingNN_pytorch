"""Tests for the Experiment 001 chart generator."""

import json

from generate_experiment_001_charts import DEFAULT_SNAPSHOT, generate_charts, load_runs


def test_load_runs_reads_model_depth_board_result(tmp_path) -> None:
    """The loader should derive one normalized row from angle/model/seed board JSON."""

    # Arrange
    board = tmp_path / "slope_2deg" / "big" / "config" / "seed_100" / "board"
    (board / "metrics").mkdir(parents=True)
    (board / "main.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "trainingTimeElapsedSec": 10,
                "generationTimeline": [
                    {
                        "generation": 1,
                        "actionExecuted": {
                            "shortLabel": "Add Res Conv Layer Action",
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
                    {"globalEpoch": 0, "trainAcc": 0.2, "valAcc": 0.2, "paramCount": 420},
                    {"globalEpoch": 1, "trainAcc": 0.9, "valAcc": 0.8, "paramCount": 1012},
                ]
            }
        ),
        encoding="utf-8",
    )

    # Act
    runs = load_runs(tmp_path)

    # Assert
    assert runs[0] | {"epochs": []} == {
        "angle": "2",
        "model": "big",
        "seed": 100,
        "status": "completed",
        "elapsed_sec": 10,
        "actions": 1,
        "action_generations": [1],
        "action_epochs": [20],
        "action_labels": ["Add Res Conv Layer Action"],
        "epochs": [],
        "final_acc": 0.8,
        "final_train_acc": 0.9,
        "best_acc": 0.8,
        "best_train_acc": 0.9,
        "final_params": 1012,
        "start_params": 420,
    }


def test_generate_charts_writes_experiment_001_figures(tmp_path) -> None:
    """The generator should create all Experiment 001 figures and a snapshot."""

    # Arrange
    runs_dir = tmp_path / "runs"
    output_dir = tmp_path / "charts"
    snapshot_path = tmp_path / "experiment-snapshot.json"
    for angle in ("2", "3", "4"):
        for model, seed, start_params in (
            ("big", 100, 420),
            ("big", 101, 420),
            ("medium", 100, 276),
            ("medium", 101, 276),
            ("very_small", 100, 76),
            ("very_small", 101, 76),
        ):
            board = (
                runs_dir
                / f"slope_{angle}deg"
                / model
                / "config"
                / f"seed_{seed}"
                / "board"
            )
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
                                        "shortLabel": "Add Res Conv Layer Action",
                                        "atGlobalEpoch": generation * 10 + 9,
                                    }
                                    if generation in (1, 3, 5)
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
    expected = {
        "001-final-accuracy.png",
        "001-final-accuracy-by-slope.png",
        "001-generation-zero.png",
        "001-param-growth.png",
        "001-representative-timeline.png",
        "001-slope-decisions.png",
        "001-actions-by-generation.png",
        "001-generation-transition.png",
        "001-action-order.png",
        "001-action-order-by-depth.png",
        "001-action-order-by-slope.png",
        "001-action-types.png",
        "001-action-types-by-depth.png",
        "001-action-composition-by-depth.png",
        "001-training-curves.png",
        "001-training-curves-by-slope.png",
    }

    # Act
    generate_charts(output_dir, runs_dir, snapshot_path)

    # Assert
    assert {path.name for path in output_dir.glob("*.png")} == expected
    assert json.loads(snapshot_path.read_text(encoding="utf-8"))[0]["model"] == "big"


def test_generate_charts_uses_snapshot_when_raw_output_is_missing(tmp_path) -> None:
    """The archived snapshot should regenerate charts without ignored raw files."""

    # Arrange
    missing_runs = tmp_path / "missing-runs"
    output_dir = tmp_path / "archived-charts"
    if not DEFAULT_SNAPSHOT.exists():
        return

    # Act
    generate_charts(output_dir, missing_runs, DEFAULT_SNAPSHOT)

    # Assert
    assert (output_dir / "001-final-accuracy.png").is_file()
    assert (output_dir / "001-action-types-by-depth.png").is_file()
