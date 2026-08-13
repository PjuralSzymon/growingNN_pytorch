"""
Experiment 004 driver — compare GrowingNN recovery-only LR vs composed base schedules.

Baseline package from Experiment 003 after_fix_1 conclusions:
- simulation grading: validation accuracy
- starter: big (strongest after-fix val_acc cell)
- slope gate: 3°
- recovery warmup: logistic, warmup_iterations=10, k=10
- epochs per generation: 10
- simulation time: 120 s

Grid factor: learning-rate schedule only.
Generations extended to 10 so global base curves have room to separate.
Three matched seeds per schedule.

Published report target:
documentation/website/content/experiments/experiment-004-composed-lr-schedulers.md
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
)
from growingnn.simulation.simulation_schedulers import SlopeEstimationSimulationScheduler
from growingnn.training.lr_scheduler_action import ActionLearningRateScheduler, LearningRateScheduler, ScheduleMode
from growingnn.training.lr_scheduler_global import (
    ComposedLearningRateScheduler,
    LinearDecayLearningRate,
    build_composed_learning_rate_scheduler,
)

RUNS_DIR = train_mnist.RUNS_DIR / "exp004_composed_lr_schedulers"
EPOCHS_PER_GENERATION = 10
GENERATIONS = 10
TOTAL_TRAINING_EPOCHS = GENERATIONS * EPOCHS_PER_GENERATION
SIMULATION_TIME_SEC = 120.0
WARMUP_ITERATIONS = 10
WARMUP_STEEPNESS = 10.0
SLOPE_ANGLE_THRESHOLD = 3.0
SCORE_ACCURACY_METRIC = "val_acc"
INITIAL_LR = 0.01
ETA_MIN = 0.001
STEP_SIZE = TOTAL_TRAINING_EPOCHS // 3
EXPONENTIAL_GAMMA = 0.98
STEP_GAMMA = 0.5
# Custom cascade: global linear base 1.0 → 0.1, times logistic recovery after actions.
CUSTOM_CASCADE_START_LR = 1.0
CUSTOM_CASCADE_END_LR = 0.1

SEED_BASE = 100
SEED_COUNT = 3
SEEDS = tuple(SEED_BASE + offset for offset in range(SEED_COUNT))

MODEL_NAME = "big"


def _factory(builder: Callable[..., torch.nn.Module], **kwargs: object) -> Callable[[dict[str, object]], torch.nn.Module]:
    def factory(_hp: dict[str, object]) -> torch.nn.Module:
        return builder(**kwargs)

    return factory


MODEL_FACTORY = _factory(
    BigAvgPoolMnistNet,
    channels=CHANNELS,
    hidden_linear_size=HIDDEN_LINEAR_SIZE,
)

# schedule_id -> human label for board metadata / charts
SCHEDULE_VARIANTS: tuple[tuple[str, str], ...] = (
    ("recovery_only_logistic", "recovery-only logistic (Exp 003 style)"),
    ("composed_cosine", "composed cosine annealing × logistic recovery"),
    ("composed_step", "composed step LR × logistic recovery"),
    ("composed_exponential", "composed exponential LR × logistic recovery"),
    ("composed_linear", "composed linear decay × logistic recovery"),
    ("composed_constant", "composed constant base × logistic recovery"),
    (
        "composed_linear_1_to_0p1",
        "custom cascade linear 1.0→0.1 × logistic recovery",
    ),
)


def build_learning_rate_scheduler_for_schedule_id(
    schedule_id: str,
    hp: dict[str, object],
):
    """
    Build a fresh scheduler for one Exp 004 cell.

    Must be called per seed so composed global_epoch state starts at 0.
    """
    initial_lr = float(hp["lr_alpha"])
    total_epochs = int(hp["generations"]) * int(hp["epochs"])

    if schedule_id == "recovery_only_logistic":
        # Absolute GrowingNN warmup only: no global base decay.
        return ActionLearningRateScheduler(
            ScheduleMode.WARMUP_LOGISTIC,
            alpha=initial_lr,
            warmup_iterations=WARMUP_ITERATIONS,
            k=WARMUP_STEEPNESS,
        )

    if schedule_id == "composed_linear_1_to_0p1":
        # User cascade: decaying peak 1.0→0.1, with low→high recovery after each action.
        return ComposedLearningRateScheduler(
            global_schedule=LinearDecayLearningRate(
                T_max=total_epochs,
                eta_min=CUSTOM_CASCADE_END_LR,
                initial_lr=CUSTOM_CASCADE_START_LR,
            ),
            recovery=ActionLearningRateScheduler(
                ScheduleMode.WARMUP_LOGISTIC,
                alpha=1.0,
                warmup_iterations=WARMUP_ITERATIONS,
                k=WARMUP_STEEPNESS,
            ),
            total_epochs=total_epochs,
            initial_lr=CUSTOM_CASCADE_START_LR,
        )

    composed_base_name = {
        "composed_cosine": "cosine",
        "composed_step": "step",
        "composed_exponential": "exponential",
        "composed_linear": "linear",
        "composed_constant": "constant",
    }.get(schedule_id)
    if composed_base_name is None:
        raise ValueError(f"Unknown Exp 004 schedule_id {schedule_id!r}")

    return build_composed_learning_rate_scheduler(
        composed_base_name,
        total_epochs=total_epochs,
        initial_lr=initial_lr,
        warmup_iterations=WARMUP_ITERATIONS,
        k=WARMUP_STEEPNESS,
        eta_min=ETA_MIN,
        step_size=STEP_SIZE,
        gamma=EXPONENTIAL_GAMMA if composed_base_name == "exponential" else STEP_GAMMA,
    )


if __name__ == "__main__":
    args = common.parse_board_cli(
        "Experiment 004: MNIST composed vs recovery-only learning-rate schedules"
    )
    configure_deterministic_seeding()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = train_mnist.MNISTData(train_mnist.DATA_DIR)
    data.prepare()
    print(f"Exp 004 write target: {RUNS_DIR}")

    for schedule_id, schedule_label in SCHEDULE_VARIANTS:
        definition = common.ExperimentDefinition(
            name=f"MNIST exp004 {schedule_id} {MODEL_NAME}",
            runs_dir=RUNS_DIR / schedule_id,
            history_filename=train_mnist.MNIST_HISTORY_FILENAME,
            seeds=SEEDS,
            folder_name=train_mnist.build_mnist_hyperparameter_folder_name,
            model_factory=MODEL_FACTORY,
            loader_factory=lambda hp: data.loaders(int(hp["batch_size"])),
            board_metadata=lambda hp, folder, seed, sid=schedule_id, label=schedule_label: (
                f"MNIST exp004 {sid} ({label}) {MODEL_NAME} | {folder} | seed {seed}",
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
            executed, skipped = common.run_experiment_grid(
                definition,
                (
                    {
                        **dict(zip(train_mnist.METAPARAM_KEYS, values)),
                        "epochs": EPOCHS_PER_GENERATION,
                        "generations": GENERATIONS,
                        "simulation_time": SIMULATION_TIME_SEC,
                        "lr_alpha": INITIAL_LR,
                        "score_accuracy_metric": SCORE_ACCURACY_METRIC,
                        "lr_scheduler_factory": (
                            lambda hp, sid=schedule_id: build_learning_rate_scheduler_for_schedule_id(
                                sid, hp
                            )
                        ),
                    }
                    for values in itertools.product(*train_mnist.METAPARAM_LISTS)
                ),
                device=device,
                board=args.board,
            )
        print(
            f"{schedule_id}: executed {executed}, skipped {skipped}, "
            f"seeds={SEEDS}, gens={GENERATIONS}, epochs={EPOCHS_PER_GENERATION}, "
            f"simt={SIMULATION_TIME_SEC}, output {definition.runs_dir}"
        )
