"""
Experiment 002 driver — initial architecture survey under fixed Exp 001 schedulers.

Experiment 001 fixed the practical scheduler pair at 3° slope + logistic LR warmup.
This follow-up keeps that pair constant and varies only the starting MNIST graph.

Research question: with sequential-convolution rebuild edges legal, which initial
architectures grow usefully under the same search/LR settings as Experiment 001?

Grid: many starters x two matched seeds (100, 101). Slope is fixed at 3°.
LR warmup is fixed to logistic. Other MNIST hyperparameters match Experiments 000/001.

Published report target:
documentation/website/content/experiments/experiment-002-initial-architectures.md

Raw output:
experiments/output/train_mnist/runs/exp002_initial_architectures
"""

from __future__ import annotations

import itertools
import sys
from functools import partial
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from unittest.mock import patch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments import experiments_common as common
from experiments import train_mnist
from experiments.train_mnist_exp001_slope_model_depth import (
    MediumMnistNet,
    VerySmallMnistNet,
    configure_deterministic_seeding,
)
from growingnn.simulation.simulation_schedulers import SlopeEstimationSimulationScheduler
from growingnn.training.lr_scheduler import LearningRateScheduler, ScheduleMode

RUNS_DIR = train_mnist.RUNS_DIR / "exp002_initial_architectures"
EPOCHS_PER_GENERATION = 10
WARMUP_ITERATIONS = 10
WARMUP_STEEPNESS = 10.0
SLOPE_ANGLE_THRESHOLD = 3.0
LR_MODE = ScheduleMode.WARMUP_LOGISTIC

SEED_BASE = 100
SEED_COUNT = 2
SEEDS = tuple(SEED_BASE + offset for offset in range(SEED_COUNT))

MNIST_SPATIAL = 28
MAX_POOL_KERNEL = 2


class MediumMaxPoolOnlyMnistNet(nn.Module):
    """Medium depth with max-pool only (no adaptive global pool)."""

    def __init__(
        self,
        num_classes: int = train_mnist.NUM_CLASSES,
        channels: int = 4,
        hidden_linear_size: int = 16,
    ) -> None:
        super().__init__()
        spatial_after_pool = MNIST_SPATIAL // MAX_POOL_KERNEL
        self.conv1 = nn.Conv2d(1, channels, 3, padding=1, bias=False)
        self.linear = nn.Linear(channels * spatial_after_pool * spatial_after_pool, hidden_linear_size, bias=True)
        self.linear2 = nn.Linear(hidden_linear_size, num_classes, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, MAX_POOL_KERNEL)
        x = F.relu(self.linear(x.flatten(1)))
        return self.linear2(x)


class MediumAvgPoolOnlyMnistNet(nn.Module):
    """Medium depth with adaptive average pool only (no max pool)."""

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
        x = F.adaptive_avg_pool2d(x, 1)
        x = F.relu(self.linear(x.flatten(1)))
        return self.linear2(x)


class MediumNoPoolMnistNet(nn.Module):
    """Medium depth with no pooling (flatten full spatial map)."""

    def __init__(
        self,
        num_classes: int = train_mnist.NUM_CLASSES,
        channels: int = 4,
        hidden_linear_size: int = 16,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, channels, 3, padding=1, bias=False)
        self.linear = nn.Linear(channels * MNIST_SPATIAL * MNIST_SPATIAL, hidden_linear_size, bias=True)
        self.linear2 = nn.Linear(hidden_linear_size, num_classes, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.linear(x.flatten(1)))
        return self.linear2(x)


def _build_big(channels: int, hidden_linear_size: int) -> Callable[[dict[str, object]], nn.Module]:
    def factory(_hp: dict[str, object]) -> nn.Module:
        return train_mnist.MinimalMnistNet(
            channels=channels,
            hidden_linear_size=hidden_linear_size,
        )

    return factory


def _build_medium(channels: int, hidden_linear_size: int) -> Callable[[dict[str, object]], nn.Module]:
    def factory(_hp: dict[str, object]) -> nn.Module:
        return MediumMnistNet(channels=channels, hidden_linear_size=hidden_linear_size)

    return factory


def _build_very_small(channels: int) -> Callable[[dict[str, object]], nn.Module]:
    def factory(_hp: dict[str, object]) -> nn.Module:
        return VerySmallMnistNet(channels=channels)

    return factory


def _build_medium_max_pool_only(
    channels: int, hidden_linear_size: int,
) -> Callable[[dict[str, object]], nn.Module]:
    def factory(_hp: dict[str, object]) -> nn.Module:
        return MediumMaxPoolOnlyMnistNet(channels=channels, hidden_linear_size=hidden_linear_size)

    return factory


def _build_medium_avg_pool_only(
    channels: int, hidden_linear_size: int,
) -> Callable[[dict[str, object]], nn.Module]:
    def factory(_hp: dict[str, object]) -> nn.Module:
        return MediumAvgPoolOnlyMnistNet(channels=channels, hidden_linear_size=hidden_linear_size)

    return factory


def _build_medium_no_pool(
    channels: int, hidden_linear_size: int,
) -> Callable[[dict[str, object]], nn.Module]:
    def factory(_hp: dict[str, object]) -> nn.Module:
        return MediumNoPoolMnistNet(channels=channels, hidden_linear_size=hidden_linear_size)

    return factory


# Starters from Experiment 001 plus proposed follow-ups (width, head, pooling).
MODEL_VARIANTS: tuple[tuple[str, Callable[[dict[str, object]], nn.Module]], ...] = (
    ("big", _build_big(4, 16)),
    ("medium", _build_medium(4, 16)),
    ("very_small", _build_very_small(4)),
    ("medium_h4", _build_medium(4, 4)),
    ("medium_ch2_h8", _build_medium(2, 8)),
    ("big_ch2_h8", _build_big(2, 8)),
    ("very_small_ch2", _build_very_small(2)),
    ("medium_max_pool_only", _build_medium_max_pool_only(4, 16)),
    ("medium_avg_pool_only", _build_medium_avg_pool_only(4, 16)),
    ("medium_no_pool", _build_medium_no_pool(4, 16)),
)


if __name__ == "__main__":
    args = common.parse_board_cli(
        "Experiment 002: fixed 3° logistic MNIST runs across initial architectures"
    )
    configure_deterministic_seeding()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = train_mnist.MNISTData(train_mnist.DATA_DIR)
    data.prepare()

    for model_name, model_factory in MODEL_VARIANTS:
        definition = common.ExperimentDefinition(
            name=f"MNIST exp002 slope_3deg logistic {model_name}",
            runs_dir=RUNS_DIR / model_name,
            history_filename=train_mnist.MNIST_HISTORY_FILENAME,
            seeds=SEEDS,
            folder_name=train_mnist.build_mnist_hyperparameter_folder_name,
            model_factory=model_factory,
            loader_factory=lambda hp: data.loaders(int(hp["batch_size"])),
            board_metadata=lambda hp, folder, seed, model=model_name: (
                f"MNIST exp002 slope_3deg logistic {model} | {folder} | seed {seed}",
                "MNIST",
            ),
        )
        with (
            patch.object(
                common,
                "AlwaysSimulationScheduler",
                partial(
                    SlopeEstimationSimulationScheduler,
                    angle_threshold=SLOPE_ANGLE_THRESHOLD,
                ),
            ),
            patch.object(
                common,
                "LearningRateScheduler",
                side_effect=lambda _mode, alpha: LearningRateScheduler(
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
            f"{model_name}: executed {executed}, skipped {skipped}, "
            f"seeds={SEEDS}, output {definition.runs_dir}"
        )
