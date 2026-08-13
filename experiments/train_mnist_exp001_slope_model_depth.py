"""
Experiment 001 driver script — continuation of Experiment 000.

Experiment 000 found that 3° slope scheduling with logistic LR warmup was the
strongest tested pair on the 420-parameter MNIST starter graph. Neuron add and
delete actions are disabled, so this follow-up shrinks capacity by removing
initial layers instead of narrowing channels.

This experiment asks whether that logistic + slope result still holds across
starting depth:

- big: Experiment 000 graph (conv1, conv2, linear, linear2; ~420 parameters)
- medium: one layer removed (drop conv2)
- very_small: two layers removed (drop conv2 and the hidden linear)

Grid: slope thresholds 2/3/4 deg x three model depths x two matched seeds.
LR warmup is fixed to logistic. Other MNIST hyperparameters match Experiment 000.

Published report target (when written):
documentation/website/content/experiments/experiment-001-slope-logistic-model-depth.md

Raw output:
experiments/output/train_mnist/runs/exp001_slope_logistic_model_depth
"""

from __future__ import annotations

import itertools
import sys
from functools import partial
from pathlib import Path
from typing import Callable
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments import experiments_common as common
from experiments import train_mnist
from growingnn.simulation.simulation_schedulers import SlopeEstimationSimulationScheduler
from growingnn.training.lr_scheduler_action import ActionLearningRateScheduler, LearningRateScheduler, ScheduleMode

RUNS_DIR = train_mnist.RUNS_DIR / "exp001_slope_logistic_model_depth"
EPOCHS_PER_GENERATION = 10
WARMUP_ITERATIONS = 10
WARMUP_STEEPNESS = 10.0
SLOPE_ANGLE_THRESHOLDS = (2.0, 3.0, 4.0)
LR_MODE = ScheduleMode.WARMUP_LOGISTIC

# Matched seeds for the grid. Values are set once here; run_experiment_grid
# applies seed_all(seed) before each model is built. Deterministic CUDA/cuDNN
# settings below replace Experiment 000's soft torch-only seeding.
SEED_BASE = 100
SEED_COUNT = 2
SEEDS = tuple(SEED_BASE + offset for offset in range(SEED_COUNT))


class MediumMnistNet(nn.Module):
    """Experiment 000 stem with conv2 removed (one fewer initial layer)."""

    def __init__(
        self,
        num_classes: int = train_mnist.NUM_CLASSES,
        channels: int = 4,
        hidden_linear_size: int = 16,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, channels, 3, padding=1, bias=False)
        self.linear = nn.Linear(channels, hidden_linear_size, bias=True)
        self.linear2 = nn.Linear(hidden_linear_size, num_classes, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.adaptive_avg_pool2d(x, 1)
        x = F.relu(self.linear(x.flatten(1)))
        return self.linear2(x)


class VerySmallMnistNet(nn.Module):
    """Experiment 000 stem with conv2 and the hidden linear removed."""

    def __init__(
        self,
        num_classes: int = train_mnist.NUM_CLASSES,
        channels: int = 4,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, channels, 3, padding=1, bias=False)
        self.linear2 = nn.Linear(channels, num_classes, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.adaptive_avg_pool2d(x, 1)
        return self.linear2(x.flatten(1))


def _build_big_model(hp: dict[str, object]) -> nn.Module:
    return train_mnist._build_model(hp)


def _build_medium_model(hp: dict[str, object]) -> nn.Module:
    return MediumMnistNet(
        channels=int(hp["model_channels"]),
        hidden_linear_size=int(hp["hidden_linear_size"]),
    )


def _build_very_small_model(hp: dict[str, object]) -> nn.Module:
    return VerySmallMnistNet(channels=int(hp["model_channels"]))


MODEL_VARIANTS: tuple[tuple[str, Callable[[dict[str, object]], nn.Module]], ...] = (
    ("big", _build_big_model),
    ("medium", _build_medium_model),
    ("very_small", _build_very_small_model),
)


def configure_deterministic_seeding() -> None:
    """Enable the Experiment 001 seeding protocol for matched seed runs."""
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    args = common.parse_board_cli(
        "Experiment 001: slope thresholds x initial MNIST model depth under logistic warmup"
    )
    configure_deterministic_seeding()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = train_mnist.MNISTData(train_mnist.DATA_DIR)
    data.prepare()

    for angle_threshold, (model_name, model_factory) in itertools.product(
        SLOPE_ANGLE_THRESHOLDS,
        MODEL_VARIANTS,
    ):
        angle_name = f"slope_{angle_threshold:g}deg"
        definition = common.ExperimentDefinition(
            name=f"MNIST exp001 {angle_name} logistic {model_name}",
            runs_dir=RUNS_DIR / angle_name / model_name,
            history_filename=train_mnist.MNIST_HISTORY_FILENAME,
            seeds=SEEDS,
            folder_name=train_mnist.build_mnist_hyperparameter_folder_name,
            model_factory=model_factory,
            loader_factory=lambda hp: data.loaders(int(hp["batch_size"])),
            board_metadata=lambda hp, folder, seed, angle=angle_name, model=model_name: (
                f"MNIST exp001 {angle} logistic {model} | {folder} | seed {seed}",
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
                    LR_MODE,
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
            f"{angle_name}/{model_name}: executed {executed}, skipped {skipped}, "
            f"seeds={SEEDS}, output {definition.runs_dir}"
        )
