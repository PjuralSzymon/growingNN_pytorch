"""
Experiment 007 driver — compare simulation-set generators on the Exp 005 keep package.

Fixed package from finished experiments (Exp 006 is unfinished, so neuron-resize stays off):
- simulation algorithm: sequential_halving_beam
- LR: composed_exponential × logistic recovery
- simulation grading: val_acc
- starter: big (BigAvgPoolMnistNet)
- slope gate: 3°
- recovery warmup: logistic, warmup_iterations=10, k=10
- generations: 10
- epochs per generation: 10
- simulation time: 120 s
- simulation set size: 2000

Grid factor: simulation-set generator only.
Three matched seeds (100, 101, 102).

Published report target:
documentation/website/content/experiments/experiment-007-simulation-sets.md

Raw output:
experiments/output/train_mnist/runs/exp007_simulation_sets
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
from growingnn.core.config import RunningConfig
from growingnn.simulation.simulation_schedulers import SlopeEstimationSimulationScheduler
from growingnn.simulation.simulation_sets import (
    CraigSimulationSet,
    El2nSimulationSet,
    GradMatchSimulationSet,
    GrandSimulationSet,
    HcdcSimulationSet,
    KCenterSimulationSet,
    ModelDriftSimulationSet,
    ModerateDifficultySimulationSet,
    ProtectedSimulationSet,
)
import growingnn.simulation.simulation_algorithms.sequential_halving_beam_alg as sequential_halving_beam_alg

_ORIGINAL_RUNNING_CONFIG = common._running_config

RUNS_DIR = train_mnist.RUNS_DIR / "exp007_simulation_sets"
SCHEDULE_ID = "composed_exponential"
SIMULATION_ALG_ID = "sequential_halving_beam"
SIMULATION_ALG = sequential_halving_beam_alg

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

SET_VARIANTS: tuple[tuple[str, Callable[[], object]], ...] = (
    ("protected", ProtectedSimulationSet),
    ("moderate_difficulty", ModerateDifficultySimulationSet),
    ("kcenter", KCenterSimulationSet),
    ("el2n", El2nSimulationSet),
    ("grand", GrandSimulationSet),
    ("grad_match", GradMatchSimulationSet),
    ("craig", CraigSimulationSet),
    ("model_drift", ModelDriftSimulationSet),
    ("hcdc", HcdcSimulationSet),
)


def _running_config_for_set(set_factory: Callable[[], object]) -> Callable[..., RunningConfig]:
    def _configure(hp, device, board) -> RunningConfig:
        cfg = _ORIGINAL_RUNNING_CONFIG(hp, device, board)
        for flag in _NEURON_RESIZE_FLAGS:
            setattr(cfg, flag, False)
        cfg.simulation_set_generator = set_factory()
        return cfg

    return _configure


def print_simulation_set_ids() -> None:
    """Print the Exp 007 simulation-set catalog and output layout."""
    print("Exp 007 simulation-set IDs:")
    for index, (set_id, factory) in enumerate(SET_VARIANTS, start=1):
        print(f"  {index:>2}. {set_id:<22} class={factory.__name__}")
    print(
        "Run path pattern: "
        f"{RUNS_DIR}/<set_id>/<hp_folder>/seed_<seed>/"
    )


if __name__ == "__main__":
    args = common.parse_board_cli(
        "Experiment 007: MNIST simulation-set generator comparison on Exp 005 keep package"
    )
    configure_deterministic_seeding()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = train_mnist.MNISTData(train_mnist.DATA_DIR)
    data.prepare()
    print(f"Exp 007 write target: {RUNS_DIR}")
    print(f"Fixed package: {SIMULATION_ALG_ID} + {SCHEDULE_ID} + {MODEL_NAME}")
    print(f"Length: gens={GENERATIONS} epochs={EPOCHS_PER_GENERATION} seeds={SEEDS}")
    print_simulation_set_ids()

    for set_id, set_factory in SET_VARIANTS:
        definition = common.ExperimentDefinition(
            name=f"MNIST exp007 {set_id} {MODEL_NAME}",
            runs_dir=RUNS_DIR / set_id,
            history_filename=train_mnist.MNIST_HISTORY_FILENAME,
            seeds=SEEDS,
            folder_name=train_mnist.build_mnist_hyperparameter_folder_name,
            model_factory=MODEL_FACTORY,
            loader_factory=lambda hp: data.loaders(int(hp["batch_size"])),
            board_metadata=lambda hp, folder, seed, sid=set_id: (
                f"MNIST exp007 {sid} ({SIMULATION_ALG_ID}/{SCHEDULE_ID}) {MODEL_NAME} | {folder} | seed {seed}",
                "MNIST",
            ),
        )
        configure = _running_config_for_set(set_factory)
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
                        "simulation_set_id": set_id,
                        "model_name": MODEL_NAME,
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
            f"set={set_id}: executed {executed}, skipped {skipped}, "
            f"seeds={SEEDS}, gens={GENERATIONS}, epochs={EPOCHS_PER_GENERATION}, "
            f"simt={SIMULATION_TIME_SEC}, output {definition.runs_dir}"
        )
