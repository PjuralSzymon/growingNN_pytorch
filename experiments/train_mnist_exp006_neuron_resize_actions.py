"""
Experiment 006 driver — short MNIST probe of neuron-resize action ratio pairs.

Fixed package from Experiment 005 / train-ci keep set:
- simulation algorithm: sequential_halving_beam
- LR: composed_exponential × logistic recovery
- simulation grading: val_acc
- starter: big (BigAvgPoolMnistNet)
- slope gate: 3°
- recovery warmup: logistic, warmup_iterations=10, k=10

Grid factor: which AddNeurons / DelNeurons ratio pair is enabled.
Layer add/delete and dropout actions stay on (RunningConfig defaults).

Four matched groups:

| ID | Enabled neuron-resize flags |
| --- | --- |
| none | all AddNeurons / DelNeurons off (Exp 001–005 style control) |
| add11_del01 | ADD_NEURONS_11 + DEL_NEURONS_01 |
| add15_del05 | ADD_NEURONS_15 + DEL_NEURONS_05 |
| add20_del09 | ADD_NEURONS_20 + DEL_NEURONS_09 |

Short run length matches the train-ci intent: 8 generations × 8 epochs.
Three matched seeds (100, 101, 102).

Published report target:
documentation/website/content/experiments/experiment-006-neuron-resize-actions.md

Raw output:
experiments/output/train_mnist/runs/exp006_neuron_resize_actions
"""

from __future__ import annotations

import itertools
import sys
from functools import partial
from pathlib import Path
from typing import Callable
from unittest.mock import patch

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments import experiments_common as common
from experiments import train_mnist
from experiments.train_mnist_exp001_slope_model_depth import configure_deterministic_seeding
from experiments.train_mnist_exp004_composed_lr_schedulers import (
    INITIAL_LR,
    MODEL_FACTORY,
    MODEL_NAME,
    SCORE_ACCURACY_METRIC,
    SIMULATION_TIME_SEC,
    SLOPE_ANGLE_THRESHOLD,
    build_learning_rate_scheduler_for_schedule_id,
)
from growingnn.core.config import RunningConfig
from growingnn.simulation.simulation_schedulers import SlopeEstimationSimulationScheduler
import growingnn.simulation.simulation_algorithms.sequential_halving_beam_alg as sequential_halving_beam_alg

_ORIGINAL_RUNNING_CONFIG = common._running_config

RUNS_DIR = train_mnist.RUNS_DIR / "exp006_neuron_resize_actions"
SCHEDULE_ID = "composed_exponential"
SIMULATION_ALG_ID = "sequential_halving_beam"
SIMULATION_ALG = sequential_halving_beam_alg

# Short probe: slightly below full Exp 005 (10×10), aligned with train-ci length intent.
EPOCHS_PER_GENERATION = 8
GENERATIONS = 8
SEED_BASE = 100
SEED_COUNT = 3
SEEDS = tuple(SEED_BASE + offset for offset in range(SEED_COUNT))

_NEURON_RESIZE_FLAGS = (
    "ACTIONS_ENABLE_ADD_NEURONS_11",
    "ACTIONS_ENABLE_ADD_NEURONS_15",
    "ACTIONS_ENABLE_ADD_NEURONS_20",
    "ACTIONS_ENABLE_DEL_NEURONS_01",
    "ACTIONS_ENABLE_DEL_NEURONS_05",
    "ACTIONS_ENABLE_DEL_NEURONS_09",
)

# (group_id, enabled_flags). Every other neuron-resize flag is forced off.
ACTION_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("none", ()),
    ("add11_del01", ("ACTIONS_ENABLE_ADD_NEURONS_11", "ACTIONS_ENABLE_DEL_NEURONS_01")),
    ("add15_del05", ("ACTIONS_ENABLE_ADD_NEURONS_15", "ACTIONS_ENABLE_DEL_NEURONS_05")),
    ("add20_del09", ("ACTIONS_ENABLE_ADD_NEURONS_20", "ACTIONS_ENABLE_DEL_NEURONS_09")),
)


def _running_config_for_group(
    enabled_flags: tuple[str, ...],
) -> Callable[..., RunningConfig]:
    enabled = set(enabled_flags)

    def _configure(hp, device, board) -> RunningConfig:
        cfg = _ORIGINAL_RUNNING_CONFIG(hp, device, board)
        for flag in _NEURON_RESIZE_FLAGS:
            setattr(cfg, flag, flag in enabled)
        return cfg

    return _configure


def print_action_groups() -> None:
    """Print the Exp 006 neuron-resize group catalog and output layout."""
    print("Exp 006 neuron-resize action groups:")
    for index, (group_id, flags) in enumerate(ACTION_GROUPS, start=1):
        enabled = ", ".join(flags) if flags else "(all neuron resize off)"
        print(f"  {index:>2}. {group_id:<14} {enabled}")
    print(
        "Run path pattern: "
        f"{RUNS_DIR}/<group_id>/<hp_folder>/seed_<seed>/"
    )


if __name__ == "__main__":
    args = common.parse_board_cli(
        "Experiment 006: MNIST short probe of AddNeurons/DelNeurons ratio pairs"
    )
    configure_deterministic_seeding()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = train_mnist.MNISTData(train_mnist.DATA_DIR)
    data.prepare()
    print(f"Exp 006 write target: {RUNS_DIR}")
    print(f"Fixed package: {SIMULATION_ALG_ID} + {SCHEDULE_ID} + {MODEL_NAME}")
    print(f"Length: gens={GENERATIONS} epochs={EPOCHS_PER_GENERATION} seeds={SEEDS}")
    print_action_groups()

    for group_id, enabled_flags in ACTION_GROUPS:
        definition = common.ExperimentDefinition(
            name=f"MNIST exp006 {group_id} {MODEL_NAME}",
            runs_dir=RUNS_DIR / group_id,
            history_filename=train_mnist.MNIST_HISTORY_FILENAME,
            seeds=SEEDS,
            folder_name=train_mnist.build_mnist_hyperparameter_folder_name,
            model_factory=MODEL_FACTORY,
            loader_factory=lambda hp: data.loaders(int(hp["batch_size"])),
            board_metadata=lambda hp, folder, seed, gid=group_id: (
                f"MNIST exp006 {gid} ({SIMULATION_ALG_ID}/{SCHEDULE_ID}) {MODEL_NAME} | {folder} | seed {seed}",
                "MNIST",
            ),
        )
        configure = _running_config_for_group(enabled_flags)
        with (
            patch.object(
                common,
                "AlwaysSimulationScheduler",
                partial(
                    SlopeEstimationSimulationScheduler,
                    angle_threshold=SLOPE_ANGLE_THRESHOLD,
                ),
            ),
            patch.object(common, "_running_config", configure),
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
                        "simulation_alg_id": SIMULATION_ALG_ID,
                        "simulation_alg": SIMULATION_ALG,
                        "model_name": MODEL_NAME,
                        "neuron_resize_group": group_id,
                        "lr_scheduler_factory": (
                            lambda hp, sid=SCHEDULE_ID: build_learning_rate_scheduler_for_schedule_id(
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
            f"group={group_id}: executed {executed}, skipped {skipped}, "
            f"seeds={SEEDS}, gens={GENERATIONS}, epochs={EPOCHS_PER_GENERATION}, "
            f"simt={SIMULATION_TIME_SEC}, output {definition.runs_dir}"
        )
