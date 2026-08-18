"""
MNIST regression CI check.

Reuse the known Experiment 004 composed_step settings on two seeds.
Print JSON metrics for the Hostinger worker. This is a gate, not an experiment.
"""

from __future__ import annotations

import itertools
import json
import sys
from functools import partial
from pathlib import Path
from unittest.mock import patch

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments import experiments_common as common
from experiments import train_mnist
from experiments.train_mnist_exp001_slope_model_depth import configure_deterministic_seeding
from experiments.train_mnist_exp004_composed_lr_schedulers import (
    EPOCHS_PER_GENERATION,
    GENERATIONS,
    INITIAL_LR,
    MODEL_FACTORY,
    MODEL_NAME,
    SCORE_ACCURACY_METRIC,
    SIMULATION_TIME_SEC,
    SLOPE_ANGLE_THRESHOLD,
    build_learning_rate_scheduler_for_schedule_id,
)
from growingnn.simulation.simulation_schedulers import SlopeEstimationSimulationScheduler

RESULT_PREFIX = "REGRESSION_CI_RESULT "
SCHEDULE_ID = "composed_step"
SEEDS = (100, 101)
DATASET = "mnist"
RUNS_DIR = _REPO_ROOT / "testResults" / "regression" / "ci" / "mnist"


def mnist_hyperparameters() -> dict[str, object]:
    """Return the known composed_step cell used by this regression check."""
    values = next(itertools.product(*train_mnist.METAPARAM_LISTS))
    return {
        **dict(zip(train_mnist.METAPARAM_KEYS, values)),
        "epochs": EPOCHS_PER_GENERATION,
        "generations": GENERATIONS,
        "simulation_time": SIMULATION_TIME_SEC,
        "lr_alpha": INITIAL_LR,
        "score_accuracy_metric": SCORE_ACCURACY_METRIC,
        "lr_scheduler_factory": (
            lambda hp: build_learning_rate_scheduler_for_schedule_id(SCHEDULE_ID, hp)
        ),
    }


def collect_metrics(runs_dir: Path, folder: str, seeds: tuple[int, ...]) -> dict[str, object]:
    """Read final val_acc and param_count for each seed from saved histories."""
    val_acc: list[float] = []
    param_count: list[int] = []
    for seed in seeds:
        history_path = runs_dir / folder / f"seed_{seed}" / train_mnist.MNIST_HISTORY_FILENAME
        history = torch.load(history_path, map_location="cpu", weights_only=False)
        val_acc.append(float(history["val_acc"][-1]))
        param_count.append(int(history["param_count"][-1]))
    return {
        "dataset": DATASET,
        "seeds": list(seeds),
        "val_acc": val_acc,
        "param_count": param_count,
    }


def result_line(payload: dict[str, object]) -> str:
    """Format the stdout contract line the Hostinger worker parses."""
    return RESULT_PREFIX + json.dumps(payload)


def run_mnist_regression(*, board: bool = False) -> dict[str, object]:
    """Train two seeds and return JSON-serializable metrics."""
    configure_deterministic_seeding()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = train_mnist.MNISTData(train_mnist.DATA_DIR)
    data.prepare()
    hp = mnist_hyperparameters()
    definition = common.ExperimentDefinition(
        name=f"MNIST regression CI {SCHEDULE_ID} {MODEL_NAME}",
        runs_dir=RUNS_DIR,
        history_filename=train_mnist.MNIST_HISTORY_FILENAME,
        seeds=SEEDS,
        folder_name=train_mnist.build_mnist_hyperparameter_folder_name,
        model_factory=MODEL_FACTORY,
        loader_factory=lambda cell: data.loaders(int(cell["batch_size"])),
        board_metadata=lambda cell, folder, seed: (
            f"MNIST regression CI {SCHEDULE_ID} {MODEL_NAME} | {folder} | seed {seed}",
            "MNIST",
        ),
    )
    with patch.object(
        common,
        "AlwaysSimulationScheduler",
        partial(
            SlopeEstimationSimulationScheduler,
            angle_threshold=SLOPE_ANGLE_THRESHOLD,
        ),
    ):
        common.run_experiment_grid(definition, (hp,), device=device, board=board)
    folder = definition.folder_name(hp)
    return collect_metrics(definition.runs_dir, folder, SEEDS)


if __name__ == "__main__":
    payload = run_mnist_regression(board=False)
    print(result_line(payload), flush=True)
