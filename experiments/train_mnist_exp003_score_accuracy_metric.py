"""
Experiment 003 driver — simulation grading metric: train accuracy vs validation accuracy.

Motivation from Experiment 002:
- Sequential convolution unlocked medium/small growth toward big.
- Dropout was over-selected and often stacked early on weak paths.
- Action choice is scored by simulation; Exp 002 used validation accuracy.
- Dropout can raise validation while training stays flat or falls, so val-based
  grading may favor regularization over learning.

Research question: under fixed 3° logistic schedulers and the two strongest Exp 002
starters, does grading simulation candidates by training accuracy reduce dropout
overuse and improve realized growth versus grading by validation accuracy?

Published report target:
documentation/website/content/experiments/experiment-003-score-accuracy-metric.md

Raw output:
experiments/output/train_mnist/runs/exp003_score_accuracy_metric
"""

from __future__ import annotations

import itertools
import sys
from functools import partial
from pathlib import Path
from typing import Callable

import torch
from unittest.mock import patch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments import experiments_common as common
from experiments import train_mnist
from experiments.train_mnist_exp001_slope_model_depth import configure_deterministic_seeding
from experiments.train_mnist_exp002_initial_architectures import (
    CHANNELS,
    HIDDEN_LINEAR_SIZE,
    BigAvgPoolMnistNet,
    Medium1Conv2LinearMnistNet,
)
from growingnn.simulation.score_functions.score_by_learning import AccuracyMetric
from growingnn.simulation.simulation_schedulers import SlopeEstimationSimulationScheduler
from growingnn.training.lr_scheduler import LearningRateScheduler, ScheduleMode

RUNS_DIR = train_mnist.RUNS_DIR / "exp003_score_accuracy_metric"
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

SCORE_METRICS: tuple[AccuracyMetric, ...] = (
    AccuracyMetric.VAL_ACC,
    AccuracyMetric.TRAIN_ACC,
)


def _factory(builder: Callable[..., torch.nn.Module], **kwargs: object) -> Callable[[dict[str, object]], torch.nn.Module]:
    def factory(_hp: dict[str, object]) -> torch.nn.Module:
        return builder(**kwargs)

    return factory


# Same two strongest topology starters from corrected Exp 002.
MODEL_VARIANTS: tuple[tuple[str, Callable[[dict[str, object]], torch.nn.Module]], ...] = (
    # 2×Conv2d + 2×Linear; start params=420
    ("big", _factory(BigAvgPoolMnistNet, channels=CHANNELS, hidden_linear_size=HIDDEN_LINEAR_SIZE)),
    # 1×Conv2d + 2×Linear; start params=276
    (
        "medium_1conv_2linear",
        _factory(
            Medium1Conv2LinearMnistNet,
            channels=CHANNELS,
            hidden_linear_size=HIDDEN_LINEAR_SIZE,
        ),
    ),
)


if __name__ == "__main__":
    args = common.parse_board_cli(
        "Experiment 003: MNIST simulation grading by train vs validation accuracy"
    )
    configure_deterministic_seeding()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = train_mnist.MNISTData(train_mnist.DATA_DIR)
    data.prepare()

    for score_metric in SCORE_METRICS:
        for model_name, model_factory in MODEL_VARIANTS:
            definition = common.ExperimentDefinition(
                name=(
                    f"MNIST exp003 slope_3deg logistic "
                    f"score_{score_metric.value} {model_name}"
                ),
                runs_dir=RUNS_DIR / score_metric.value / model_name,
                history_filename=train_mnist.MNIST_HISTORY_FILENAME,
                seeds=SEEDS,
                folder_name=train_mnist.build_mnist_hyperparameter_folder_name,
                model_factory=model_factory,
                loader_factory=lambda hp: data.loaders(int(hp["batch_size"])),
                board_metadata=lambda hp, folder, seed, metric=score_metric.value, model=model_name: (
                    f"MNIST exp003 score_{metric} {model} | {folder} | seed {seed}",
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
                            "generations": GENERATIONS,
                            "simulation_time": SIMULATION_TIME_SEC,
                            "score_accuracy_metric": score_metric.value,
                        }
                        for values in itertools.product(*train_mnist.METAPARAM_LISTS)
                    ),
                    device=device,
                    board=args.board,
                )
            print(
                f"{score_metric.value}/{model_name}: executed {executed}, skipped {skipped}, "
                f"seeds={SEEDS}, gens={GENERATIONS}, simt={SIMULATION_TIME_SEC}, "
                f"output {definition.runs_dir}"
            )
