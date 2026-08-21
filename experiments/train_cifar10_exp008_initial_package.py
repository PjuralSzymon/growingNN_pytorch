"""
Experiment 008 driver — CIFAR-10 adaptation of the finished Experiment 005 package.

Fixed package from Experiment 005 (006 and 007 are unfinished, so they stay out):
- simulation algorithm: sequential_halving_beam
- LR: composed_exponential × logistic recovery
- simulation grading: val_acc
- slope gate: 3°
- recovery warmup: logistic, warmup_iterations=10, k=10
- generations: 10
- epochs per generation: 10 (one variant uses 20)
- simulation time: 120 s
- neuron-resize flags: off

Grid factor: one CIFAR-specific change per variant around a single base cell.
Three matched seeds (100, 101, 102).

Published report target:
documentation/website/content/experiments/experiment-008-cifar10-initial-package.md

Raw output:
experiments/output/train_cifar10/runs/exp008_cifar10_initial_package

Smoke: python experiments/train_cifar10_exp008_initial_package.py --variant base --seeds 100
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from unittest.mock import patch

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments import experiments_common as common
from experiments import train_cifar10
from experiments.createsummary import HISTORY_FILENAME, build_hyperparameter_folder_name
from experiments.train_mnist_exp001_slope_model_depth import configure_deterministic_seeding
from experiments.train_mnist_exp004_composed_lr_schedulers import (
    EPOCHS_PER_GENERATION,
    GENERATIONS,
    INITIAL_LR,
    SCORE_ACCURACY_METRIC,
    SIMULATION_TIME_SEC,
    SLOPE_ANGLE_THRESHOLD,
    build_learning_rate_scheduler_for_schedule_id,
)
from growingnn.core.config import RunningConfig
import growingnn.core.config as growingnn_config
from growingnn.simulation.simulation_schedulers import (
    AlwaysSimulationScheduler,
    NeverSimulationScheduler,
    SlopeEstimationSimulationScheduler,
)
import growingnn.simulation.simulation_algorithms.sequential_halving_beam_alg as sequential_halving_beam_alg

_ORIGINAL_RUNNING_CONFIG = common._running_config

RUNS_DIR = train_cifar10.RUNS_DIR / "exp008_cifar10_initial_package"
SCHEDULE_ID = "composed_exponential"
SIMULATION_ALG_ID = "sequential_halving_beam"
SIMULATION_ALG = sequential_halving_beam_alg
RESIDUAL_CONV_POOL_TYPE = "avg"
BATCH_SIZE = 64
SIMULATION_EPOCHS = 15
SIMULATION_SET_SIZE = 2000
TARGET_ACCURACY = 0.99
SCORE_WEIGHT_ACC = 1.0
SCORE_WEIGHT_COUNTW = 0.1
EPOCHS_20 = 20
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

SCHEDULER_SLOPE = "slope"
SCHEDULER_ALWAYS = "always"
SCHEDULER_NEVER = "never"


@dataclass(frozen=True)
class CifarPackageVariant:
    variant_id: str
    channels: int
    hidden_dim: int
    epochs: int
    scheduler: str


VARIANTS: tuple[CifarPackageVariant, ...] = (
    CifarPackageVariant("narrow", 4, 32, EPOCHS_PER_GENERATION, SCHEDULER_SLOPE),  # 33390 params
    CifarPackageVariant("base", 8, 38, EPOCHS_PER_GENERATION, SCHEDULER_SLOPE),  # 79060 params
    CifarPackageVariant("deep", 16, 48, EPOCHS_PER_GENERATION, SCHEDULER_SLOPE),  # 199914 params
    CifarPackageVariant("epochs20", 8, 38, EPOCHS_20, SCHEDULER_SLOPE),  # 79060 params
    CifarPackageVariant("always", 8, 38, EPOCHS_PER_GENERATION, SCHEDULER_ALWAYS),  # 79060 params
    CifarPackageVariant("fixed", 8, 38, EPOCHS_PER_GENERATION, SCHEDULER_NEVER),  # 79060 params
)


def variant_by_id(variant_id: str) -> CifarPackageVariant:
    """Return the catalog entry for one Exp 008 variant id."""
    for variant in VARIANTS:
        if variant.variant_id == variant_id:
            return variant
    raise KeyError(f"unknown Exp 008 variant {variant_id!r}")


def apply_residual_conv_pool_patch():
    """Force residual-into-linear skips to use average pool when growth adds them."""
    return patch.object(
        growingnn_config,
        "RES_CONV_TO_LINEAR_GLOBAL_POOL_TYPE",
        RESIDUAL_CONV_POOL_TYPE,
    )


def hyperparameters_for_variant(variant: CifarPackageVariant) -> dict[str, object]:
    """Build the Exp 005 package hyperparameters with one CIFAR variant overlay."""
    return {
        "generations": GENERATIONS,
        "epochs": variant.epochs,
        "batch_size": BATCH_SIZE,
        "lr_alpha": INITIAL_LR,
        "simulation_time": SIMULATION_TIME_SEC,
        "simulation_epochs": SIMULATION_EPOCHS,
        "simulation_set_size": SIMULATION_SET_SIZE,
        "target_accuracy": TARGET_ACCURACY,
        "score_weight_acc": SCORE_WEIGHT_ACC,
        "score_weight_countw": SCORE_WEIGHT_COUNTW,
        "model_channels": variant.channels,
        "model_hidden_dim": variant.hidden_dim,
        "score_accuracy_metric": SCORE_ACCURACY_METRIC,
        "simulation_alg_id": SIMULATION_ALG_ID,
        "simulation_alg": SIMULATION_ALG,
        "variant_id": variant.variant_id,
        "scheduler_id": variant.scheduler,
        "lr_scheduler_factory": (
            lambda hp, sid=SCHEDULE_ID: build_learning_rate_scheduler_for_schedule_id(sid, hp)
        ),
    }


def _scheduler_for_kind(kind: str, hp: dict[str, object]):
    time_s = float(hp["simulation_time"])
    sim_epochs = int(hp["simulation_epochs"])
    if kind == SCHEDULER_NEVER:
        return NeverSimulationScheduler(time_s, sim_epochs)
    if kind == SCHEDULER_ALWAYS:
        return AlwaysSimulationScheduler(time_s, sim_epochs)
    if kind == SCHEDULER_SLOPE:
        return SlopeEstimationSimulationScheduler(
            time_s,
            sim_epochs,
            angle_threshold=SLOPE_ANGLE_THRESHOLD,
        )
    raise ValueError(f"unknown Exp 008 scheduler kind {kind!r}")


def running_config_for_variant(scheduler_kind: str) -> Callable[..., RunningConfig]:
    """Build RunningConfig with neuron-resize off and the variant simulation gate."""

    def _configure(hp: dict[str, object], device: torch.device, board) -> RunningConfig:
        cfg = _ORIGINAL_RUNNING_CONFIG(hp, device, board)
        for flag in _NEURON_RESIZE_FLAGS:
            setattr(cfg, flag, False)
        cfg.simulation_scheduler = _scheduler_for_kind(scheduler_kind, hp)
        return cfg

    return _configure


def parse_exp008_cli() -> argparse.Namespace:
    """Parse board output plus optional smoke filters for variant and seeds."""
    parser = argparse.ArgumentParser(
        description="Experiment 008: CIFAR-10 one-factor adaptation of the Exp 005 package"
    )
    parser.add_argument(
        "--board",
        choices=("true", "false"),
        default="true",
        help="Write GrowingNN Board artifacts under each run's board/ folder (default: true)",
    )
    parser.add_argument(
        "--variant",
        action="append",
        choices=[variant.variant_id for variant in VARIANTS],
        help="Run only this variant id. Repeat to select several. Default: all six.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Run only these seeds. Default: 100 101 102. Smoke: --variant base --seeds 100",
    )
    args = parser.parse_args()
    args.board = args.board == "true"
    return args


def print_variants() -> None:
    """Print the Exp 008 variant catalog, frozen package, and output layout."""
    print("Exp 008 CIFAR-10 package variants (one factor changes from base):")
    for index, variant in enumerate(VARIANTS, start=1):
        print(
            f"  {index:>2}. {variant.variant_id:<10} "
            f"ch={variant.channels} hd={variant.hidden_dim} "
            f"epochs={variant.epochs} scheduler={variant.scheduler}"
        )
    print(f"Fixed package: {SIMULATION_ALG_ID} + {SCHEDULE_ID} + slope {SLOPE_ANGLE_THRESHOLD}deg")
    print(
        "Run path pattern: "
        f"{RUNS_DIR}/<variant_id>/<hp_folder>/seed_<seed>/"
    )
    print("Smoke: python experiments/train_cifar10_exp008_initial_package.py --variant base --seeds 100")


if __name__ == "__main__":
    args = parse_exp008_cli()
    configure_deterministic_seeding()
    device = torch.device("cuda")
    common.require_cuda(device)
    data = train_cifar10.Cifar10Data(train_cifar10.DATA_DIR)
    data.prepare()
    selected_ids = tuple(args.variant) if args.variant else tuple(v.variant_id for v in VARIANTS)
    selected_seeds = tuple(args.seeds) if args.seeds else SEEDS
    print(f"Exp 008 write target: {RUNS_DIR}")
    print(f"Selected variants={selected_ids} seeds={selected_seeds}")
    print_variants()

    with apply_residual_conv_pool_patch():
        for variant_id in selected_ids:
            variant = variant_by_id(variant_id)
            hp = hyperparameters_for_variant(variant)
            definition = common.ExperimentDefinition(
                name=f"CIFAR-10 exp008 {variant.variant_id}",
                runs_dir=RUNS_DIR / variant.variant_id,
                history_filename=HISTORY_FILENAME,
                seeds=selected_seeds,
                folder_name=build_hyperparameter_folder_name,
                model_factory=train_cifar10._build_model,
                loader_factory=lambda cell: data.loaders(int(cell["batch_size"])),
                board_metadata=lambda _hp, folder, seed, vid=variant.variant_id: (
                    f"CIFAR-10 exp008 {vid} ({SIMULATION_ALG_ID}/{SCHEDULE_ID}) | {folder} | seed {seed}",
                    "CIFAR-10",
                ),
            )
            configure = running_config_for_variant(variant.scheduler)
            with patch.object(common, "_running_config", configure):
                executed, skipped = common.run_experiment_grid(
                    definition,
                    (hp,),
                    device=device,
                    board=args.board,
                )
            print(
                f"variant={variant.variant_id}: executed {executed}, skipped {skipped}, "
                f"seeds={selected_seeds}, gens={GENERATIONS}, epochs={variant.epochs}, "
                f"simt={SIMULATION_TIME_SEC}, scheduler={variant.scheduler}, "
                f"output {definition.runs_dir}"
            )
