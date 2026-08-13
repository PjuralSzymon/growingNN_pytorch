"""
Experiment 000 driver script.

This is the source script for the published website experiment:
documentation/website/content/experiments/experiment-000-slope-angle-lr-warmup.md

It produces:
experiments/output/train_mnist/runs/lr_scheduler_slope_angle_experiment

Grid: slope thresholds 1/3 deg x cosine/logistic/exponential warmup x seeds 1/2.
"""

from __future__ import annotations

import itertools
from functools import partial
from unittest.mock import patch

import torch
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments import experiments_common as common
from experiments import train_mnist
from growingnn.simulation.simulation_schedulers import SlopeEstimationSimulationScheduler
from growingnn.training.lr_scheduler_action import ActionLearningRateScheduler, LearningRateScheduler, ScheduleMode

RUNS_DIR = train_mnist.RUNS_DIR / "lr_scheduler_slope_angle_experiment"
EPOCHS_PER_GENERATION = 10
WARMUP_ITERATIONS = 10
WARMUP_STEEPNESS = 10.0
SLOPE_ANGLE_THRESHOLDS = (1.0, 3.0)
SEEDS = (1, 2)
LR_SCHEDULERS = (
    ScheduleMode.WARMUP_COSINE,
    ScheduleMode.WARMUP_LOGISTIC,
    ScheduleMode.WARMUP_EXPONENTIAL,
)


if __name__ == "__main__":
    args = common.parse_board_cli("Compare warmup LR schedulers on train_mnist")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = train_mnist.MNISTData(train_mnist.DATA_DIR)
    data.prepare()

    for angle_threshold, mode in itertools.product(
        SLOPE_ANGLE_THRESHOLDS,
        LR_SCHEDULERS,
    ):
        scheduler_name = mode.name.lower()
        angle_name = f"slope_{angle_threshold:g}deg"
        definition = common.ExperimentDefinition(
            name=f"MNIST {angle_name} {scheduler_name}",
            runs_dir=RUNS_DIR / angle_name / scheduler_name,
            history_filename=train_mnist.MNIST_HISTORY_FILENAME,
            seeds=SEEDS,
            folder_name=train_mnist.build_mnist_hyperparameter_folder_name,
            model_factory=train_mnist._build_model,
            loader_factory=lambda hp: data.loaders(int(hp["batch_size"])),
            board_metadata=lambda hp, folder, seed: (
                f"MNIST {angle_name} {scheduler_name} | {folder} | seed {seed}",
                "MNIST",
            ),
        )
        with (
            patch.object(
                common,
                "AlwaysSimulationScheduler",
                partial(
                    SlopeEstimationSimulationScheduler,
                    angle_threshold=angle_threshold,
                ),
            ),
            patch.object(common, "ActionLearningRateScheduler",
                side_effect=lambda _mode, alpha: ActionLearningRateScheduler(
                    mode,
                    alpha,
                    warmup_iterations=WARMUP_ITERATIONS,
                    k=WARMUP_STEEPNESS,
                ),
            ),
        ):
            executed, skipped = common.run_experiment_grid(
                definition,
                (
                    {
                        **dict(zip(train_mnist.METAPARAM_KEYS, values)),
                        "epochs": EPOCHS_PER_GENERATION,
                    }
                    for values in itertools.product(*train_mnist.METAPARAM_LISTS)
                ),
                device=device,
                board=args.board,
            )
        print(
            f"{angle_name}/{scheduler_name}: executed {executed}, skipped {skipped}, "
            f"output {definition.runs_dir}"
        )
