"""Tests for the focused experiment chart generator."""

import json

from generate_experiment_charts import DEFAULT_SNAPSHOT, generate_charts, load_runs


def test_load_runs_reads_board_result(tmp_path) -> None:
    """The loader should derive one normalized row from board JSON."""

    # Arrange
    board = tmp_path / "slope_1deg" / "warmup_cosine" / "config" / "seed_1" / "board"
    (board / "metrics").mkdir(parents=True)
    (board / "main.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "trainingTimeElapsedSec": 10,
                "generationTimeline": [
                    {
                        "generation": 0,
                        "actionExecuted": {
                            "shortLabel": "Add Seq Linear Layer Action",
                            "atGlobalEpoch": 1,
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
                    {"globalEpoch": 0, "trainAcc": 0.6, "valAcc": 0.5, "paramCount": 420},
                    {"globalEpoch": 1, "trainAcc": 0.9, "valAcc": 0.8, "paramCount": 500},
                ]
            }
        ),
        encoding="utf-8",
    )

    # Act
    runs = load_runs(tmp_path)

    # Assert
    assert runs[0] | {"epochs": []} == {
        "angle": "1",
        "mode": "cosine",
        "seed": 1,
        "status": "completed",
        "elapsed_sec": 10,
        "actions": 1,
        "action_generations": [0],
        "action_epochs": [1],
        "action_labels": ["Add Seq Linear Layer Action"],
        "epochs": [],
        "final_acc": 0.8,
        "final_train_acc": 0.9,
        "best_acc": 0.8,
        "best_train_acc": 0.9,
        "final_params": 500,
    }


def test_generate_charts_writes_clear_experiment_figures(tmp_path) -> None:
    """The generator should create all focused experiment figures."""

    # Arrange
    runs_dir = tmp_path / "runs"
    board = runs_dir / "slope_3deg" / "warmup_logistic" / "config" / "seed_1" / "board"
    output_dir = tmp_path / "charts"
    snapshot_path = tmp_path / "experiment-snapshot.json"
    epochs = [
        {
            "globalEpoch": generation * 2 + epoch,
            "generation": generation,
            "trainAcc": 0.5 + generation * 0.02 + epoch * 0.01,
            "valAcc": 0.5 + generation * 0.02 + epoch * 0.01,
            "lr": 0.001 if generation in (1, 3) and epoch == 0 else 0.01,
            "paramCount": 500,
        }
        for generation in range(10)
        for epoch in range(2)
    ]
    (board / "metrics").mkdir(parents=True)
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
                                "shortLabel": "Add Seq Linear Layer Action",
                                "atGlobalEpoch": generation * 2 + 1,
                            }
                            if generation in (0, 2)
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
        "000-final-accuracy.png",
        "000-learning-rate-design.png",
        "000-instability-risk.png",
        "000-representative-timeline.png",
        "000-slope-decisions.png",
        "000-generation-transition.png",
        "000-signed-generation-transition.png",
        "000-action-order.png",
        "000-action-types.png",
        "000-actions-by-generation.png",
        "000-training-curves.png",
    }

    # Act
    generate_charts(output_dir, runs_dir, snapshot_path)

    # Assert
    assert {path.name for path in output_dir.glob("*.png")} == expected
    assert json.loads(snapshot_path.read_text(encoding="utf-8"))[0]["mode"] == "logistic"


def test_generate_charts_uses_snapshot_when_raw_output_is_missing(tmp_path) -> None:
    """The archived snapshot should regenerate charts without ignored raw files."""

    # Arrange
    missing_runs = tmp_path / "missing-runs"
    output_dir = tmp_path / "archived-charts"

    # Act
    generate_charts(output_dir, missing_runs, DEFAULT_SNAPSHOT)

    # Assert
    assert (output_dir / "000-final-accuracy.png").is_file()
