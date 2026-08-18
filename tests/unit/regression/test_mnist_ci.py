"""Unit tests for the MNIST regression CI check constants and metric collection."""

import sys
from pathlib import Path

import torch

_CI_DIR = Path(__file__).resolve().parents[2] / "regression" / "ci"
if str(_CI_DIR) not in sys.path:
    sys.path.insert(0, str(_CI_DIR))

import mnist as mnist_job
from experiments.train_mnist_exp004_composed_lr_schedulers import (
    EPOCHS_PER_GENERATION,
    GENERATIONS,
    INITIAL_LR,
    SCORE_ACCURACY_METRIC,
    SIMULATION_TIME_SEC,
    SLOPE_ANGLE_THRESHOLD,
)


def test_mnist_regression_job_reuses_exp004_composed_step_two_seeds():
    """
    The CI check should train the known composed_step cell on seeds 100 and 101.
    """

    # Arrange / Act / Assert
    assert mnist_job.SCHEDULE_ID == "composed_step"
    assert mnist_job.SEEDS == (100, 101)
    assert mnist_job.DATASET == "mnist"
    assert mnist_job.GENERATIONS == GENERATIONS
    assert mnist_job.EPOCHS_PER_GENERATION == EPOCHS_PER_GENERATION
    assert mnist_job.SIMULATION_TIME_SEC == SIMULATION_TIME_SEC
    assert mnist_job.INITIAL_LR == INITIAL_LR
    assert mnist_job.SCORE_ACCURACY_METRIC == SCORE_ACCURACY_METRIC
    assert mnist_job.SLOPE_ANGLE_THRESHOLD == SLOPE_ANGLE_THRESHOLD


def test_mnist_hyperparameters_override_exp004_cell_fields():
    """
    mnist_hyperparameters should keep the known composed_step training budget.
    """

    # Arrange / Act
    hp = mnist_job.mnist_hyperparameters()

    # Assert
    assert hp["epochs"] == EPOCHS_PER_GENERATION
    assert hp["generations"] == GENERATIONS
    assert hp["simulation_time"] == SIMULATION_TIME_SEC
    assert hp["lr_alpha"] == INITIAL_LR
    assert hp["score_accuracy_metric"] == SCORE_ACCURACY_METRIC
    assert callable(hp["lr_scheduler_factory"])


def test_collect_metrics_reads_final_val_acc_and_param_count(tmp_path: Path):
    """
    collect_metrics should return the last val_acc and param_count for each seed.
    """

    # Arrange
    folder = "cell"
    for seed, acc, params in ((100, 0.879, 1200), (101, 0.921, 1300)):
        run_dir = tmp_path / folder / f"seed_{seed}"
        run_dir.mkdir(parents=True)
        torch.save(
            {"val_acc": [0.1, acc], "param_count": [420, params]},
            run_dir / "train_mnist_history.pt",
        )

    # Act
    payload = mnist_job.collect_metrics(tmp_path, folder, (100, 101))

    # Assert
    assert payload == {
        "dataset": "mnist",
        "seeds": [100, 101],
        "val_acc": [0.879, 0.921],
        "param_count": [1200, 1300],
    }


def test_result_line_prefixes_json_for_the_worker():
    """
    result_line should emit the stdout contract the Hostinger worker parses.
    """

    # Arrange
    payload = {"dataset": "mnist", "seeds": [100]}

    # Act
    line = mnist_job.result_line(payload)

    # Assert
    assert line == 'REGRESSION_CI_RESULT {"dataset": "mnist", "seeds": [100]}'
