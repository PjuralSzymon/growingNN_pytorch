"""
Experiment 002 driver — initial architecture survey (revised after report analysis).

Lessons from the first Exp 002 grid:
- Do not stack pooling ops next to each other.
- Keep one pooling style, and keep the first linear compact via global pooling.
- Compare topology only: same channels, same hidden size, same pooling for every starter.
- Fewer generations and a shorter MCTS budget reduce late random actions.

Research question: with sequential-convolution rebuild edges legal, which starting
layer layouts grow usefully under fixed 3° logistic schedulers when width is held fixed?

Published report target:
documentation/website/content/experiments/experiment-002-initial-architectures.md

Raw output (published corrected grid):
experiments/output/train_mnist/runs/exp002_initial_architectures_after_fix_1
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
from experiments.train_mnist_exp001_slope_model_depth import configure_deterministic_seeding
from growingnn.simulation.simulation_schedulers import SlopeEstimationSimulationScheduler
from growingnn.training.lr_scheduler_action import ActionLearningRateScheduler, LearningRateScheduler, ScheduleMode

RUNS_DIR = train_mnist.RUNS_DIR / "exp002_initial_architectures_after_fix_1"
EPOCHS_PER_GENERATION = 10
GENERATIONS = 5
SIMULATION_TIME_SEC = 120.0
WARMUP_ITERATIONS = 10
WARMUP_STEEPNESS = 10.0
SLOPE_ANGLE_THRESHOLD = 3.0
LR_MODE = ScheduleMode.WARMUP_LOGISTIC

SEED_BASE = 100
SEED_COUNT = 4
SEEDS = tuple(SEED_BASE + offset for offset in range(SEED_COUNT))

# Shared width for every starter. Topology is the only intentional difference.
CHANNELS = 4
HIDDEN_LINEAR_SIZE = 16


class BigAvgPoolMnistNet(nn.Module):
    """2×Conv2d + 2×Linear; one adaptive average pool before the hidden linear."""

    def __init__(
        self,
        num_classes: int = train_mnist.NUM_CLASSES,
        channels: int = CHANNELS,
        hidden_linear_size: int = HIDDEN_LINEAR_SIZE,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.linear = nn.Linear(channels, hidden_linear_size, bias=True)
        self.linear2 = nn.Linear(hidden_linear_size, num_classes, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.adaptive_avg_pool2d(x, 1)
        x = F.relu(self.linear(x.flatten(1)))
        return self.linear2(x)


class Medium1Conv2LinearMnistNet(nn.Module):
    """1×Conv2d + 2×Linear; one adaptive average pool before the hidden linear."""

    def __init__(
        self,
        num_classes: int = train_mnist.NUM_CLASSES,
        channels: int = CHANNELS,
        hidden_linear_size: int = HIDDEN_LINEAR_SIZE,
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


class Medium2Conv1LinearMnistNet(nn.Module):
    """2×Conv2d + 1×Linear; one adaptive average pool before the classifier linear."""

    def __init__(
        self,
        num_classes: int = train_mnist.NUM_CLASSES,
        channels: int = CHANNELS,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.linear2 = nn.Linear(channels, num_classes, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.adaptive_avg_pool2d(x, 1)
        return self.linear2(x.flatten(1))


class SmallAvgPoolMnistNet(nn.Module):
    """1×Conv2d + 1×Linear; one adaptive average pool before the classifier linear."""

    def __init__(
        self,
        num_classes: int = train_mnist.NUM_CLASSES,
        channels: int = CHANNELS,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, channels, 3, padding=1, bias=False)
        self.linear2 = nn.Linear(channels, num_classes, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.adaptive_avg_pool2d(x, 1)
        return self.linear2(x.flatten(1))


def _factory(builder: Callable[..., nn.Module], **kwargs: object) -> Callable[[dict[str, object]], nn.Module]:
    def factory(_hp: dict[str, object]) -> nn.Module:
        return builder(**kwargs)

    return factory


# Topology-only grid. Width and pooling are shared; rows are ordered by start params, largest first.
# Param counts with channels=4, hidden=16, classes=10:
#   big                    2×Conv+2×Linear → 420
#   medium_1conv_2linear   1×Conv+2×Linear → 276
#   medium_2conv_1linear   2×Conv+1×Linear → 220
#   small                  1×Conv+1×Linear → 76
MODEL_VARIANTS: tuple[tuple[str, Callable[[dict[str, object]], nn.Module]], ...] = (
    # 2×Conv2d + 2×Linear (4 modules); adaptive_avg only; channels=4, hidden=16; start params=420
    ("big", _factory(BigAvgPoolMnistNet)),
    # 1×Conv2d + 2×Linear (3 modules); adaptive_avg only; channels=4, hidden=16; start params=276
    ("medium_1conv_2linear", _factory(Medium1Conv2LinearMnistNet)),
    # 2×Conv2d + 1×Linear (3 modules); adaptive_avg only; channels=4; start params=220
    ("medium_2conv_1linear", _factory(Medium2Conv1LinearMnistNet)),
    # 1×Conv2d + 1×Linear (2 modules); adaptive_avg only; channels=4; start params=76
    ("small", _factory(SmallAvgPoolMnistNet)),
)


if __name__ == "__main__":
    args = common.parse_board_cli(
        "Experiment 002: MNIST initial-architecture survey (topology-only, avg pool)"
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
                        "generations": GENERATIONS,
                        "simulation_time": SIMULATION_TIME_SEC,
                    }
                    for values in itertools.product(*train_mnist.METAPARAM_LISTS)
                ),
                device=device,
                board=args.board,
            )
        print(
            f"{model_name}: executed {executed}, skipped {skipped}, "
            f"seeds={SEEDS}, gens={GENERATIONS}, simt={SIMULATION_TIME_SEC}, "
            f"output {definition.runs_dir}"
        )
