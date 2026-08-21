"""Tests for the Experiment 008 CIFAR-10 chart generator."""

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from generate_experiment_008_charts import (
    DEFAULT_RUNS,
    DEFAULT_SNAPSHOT,
    GROUP_ORDER,
    generate_charts,
    load_runs,
)


def test_default_runs_point_at_cifar10_exp008_folder():
    """
    Published Exp 008 charts should read the CIFAR-10 exp008 output root.
    """

    # Arrange / Act / Assert
    assert DEFAULT_RUNS.name == "exp008_cifar10_initial_package"
    assert DEFAULT_RUNS.parent.name == "runs"
    assert DEFAULT_SNAPSHOT.name == "experiment-008-cifar10-initial-package.json"
    assert GROUP_ORDER == ("base", "narrow", "deep", "epochs20", "always", "fixed")


def test_load_runs_reads_variant_board_and_simulation_count(tmp_path: Path):
    """
    The loader should derive one normalized row from variant/seed board JSON.
    """

    # Arrange
    board = tmp_path / "base" / "config" / "seed_100" / "board"
    (board / "metrics").mkdir(parents=True)
    (board / "simulations").mkdir(parents=True)
    (board / "main.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "trainingParameters": {"epochsPerGeneration": 10},
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
    (board / "simulations" / "simulation_gen_1.json").write_text("{}", encoding="utf-8")
    epochs = []
    for generation in range(3):
        for local in range(10):
            epochs.append(
                {
                    "globalEpoch": generation * 10 + local,
                    "generation": generation,
                    "trainAcc": 0.2 + 0.01 * (generation * 10 + local),
                    "valAcc": 0.25 + 0.01 * (generation * 10 + local),
                    "paramCount": 1000 + generation,
                }
            )
    (board / "metrics" / "training.json").write_text(
        json.dumps({"epochs": epochs}),
        encoding="utf-8",
    )

    # Act
    runs = load_runs(tmp_path)

    # Assert
    assert len(runs) == 1
    assert runs[0]["group_id"] == "base"
    assert runs[0]["seed"] == 100
    assert runs[0]["simulations_ran"] == 1
    assert runs[0]["actions"] == 1
    assert runs[0]["action_labels"] == ["Add Res Conv Layer Action"]
    assert runs[0]["epochs_per_generation"] == 10
    assert runs[0]["post_action_train_changes"]


def test_generate_charts_writes_exp008_figure_files(tmp_path: Path):
    """
    generate_charts should write the seven Exp 008 figure filenames.
    """

    # Arrange
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    runs = [
        {
            "group_id": "base",
            "seed": 100,
            "status": "completed",
            "simulations_ran": 2,
            "actions": 1,
            "action_generations": [1],
            "action_labels": ["Add Res Conv Layer Action"],
            "epochs_per_generation": 10,
            "train_acc": [0.2 + 0.01 * i for i in range(20)],
            "val_acc": [0.25 + 0.01 * i for i in range(20)],
            "param_count": [1000] * 20,
            "final_train_acc": 0.4,
            "final_val_acc": 0.45,
            "start_params": 1000,
            "final_params": 1200,
            "immediate_post_action_train_changes": [1.0],
            "post_action_train_changes": [3.0],
        }
    ]

    # Act
    generate_charts(runs, output_dir)

    # Assert
    expected = (
        "008-final-accuracy-by-variant.png",
        "008-param-growth-by-variant.png",
        "008-action-composition-by-variant.png",
        "008-search-activity-by-variant.png",
        "008-post-action-recovery-by-variant.png",
        "008-training-curves.png",
        "008-validation-curves.png",
    )
    for name in expected:
        assert (output_dir / name).is_file(), name
