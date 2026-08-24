"""
Experiment 008 driver — CIFAR-10 adaptive meta-parameter search (issue #73).

The search algorithm lives in experiments/adaptive_metaparameter_search.py.
This file only lists groups, maps a combo to a run, and loops next_config/record.

Published report:
documentation/website/content/experiments/experiment-008-cifar10-initial-package.md

Raw output:
experiments/output/train_cifar10/runs/exp008_cifar10_initial_package

Live status (refresh while running):
experiments/output/train_cifar10/runs/exp008_cifar10_initial_package/adaptive_search.md

Smoke: python experiments/train_cifar10_exp008_initial_package.py --max-iters 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest.mock import patch

import torch
import torch.fx as fx

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments import experiments_common as common
from experiments.adaptive_metaparameter_search import AdaptiveMetaParameterSearch
from experiments import train_cifar10
from experiments.createsummary import HISTORY_FILENAME
from experiments.train_mnist_exp001_slope_model_depth import configure_deterministic_seeding
from experiments.train_mnist_exp004_composed_lr_schedulers import (
    ETA_MIN,
    EXPONENTIAL_GAMMA,
    STEP_GAMMA,
    WARMUP_ITERATIONS,
    WARMUP_STEEPNESS,
)
from growingnn.board import ExperimentBoard
from growingnn.core.config import RunningConfig
import growingnn.core.config as growingnn_config
from growingnn.core.logger import logger
from growingnn.simulation.simulation_schedulers import (
    AlwaysSimulationScheduler,
    SlopeEstimationSimulationScheduler,
)
import growingnn.simulation.simulation_algorithms.best_first_alg as best_first_alg
import growingnn.simulation.simulation_algorithms.greedy_alg as greedy_alg
import growingnn.simulation.simulation_algorithms.montecarlo_alg as montecarlo_alg
import growingnn.simulation.simulation_algorithms.sequential_halving_beam_alg as sequential_halving_beam_alg
import growingnn.simulation.simulation_algorithms.ugape_deepen_alg as ugape_deepen_alg
from growingnn.training.lr_scheduler_global import build_composed_learning_rate_scheduler
from growingnn.training.trainer import train_generations
from growingnn.utils.fx import GraphStructureQuery
from growingnn.utils.seed import seed_all

RUNS_DIR = train_cifar10.RUNS_DIR / "exp008_cifar10_initial_package"
RESIDUAL_CONV_POOL_TYPE = "avg"
BATCH_SIZE = 64
TARGET_ACCURACY = 0.99
SCORE_WEIGHT_ACC = 1.0
SCORE_WEIGHT_COUNTW = 0.1
SCORE_ACCURACY_METRIC = "val_acc"
SEED_BASE = 100
MAX_ITERS = 50
N_INIT = 5
TAU = 0.15
BETA = 0.3

_NEURON_RESIZE_FLAGS = (
    "ACTIONS_ENABLE_ADD_NEURONS_11",
    "ACTIONS_ENABLE_ADD_NEURONS_15",
    "ACTIONS_ENABLE_ADD_NEURONS_20",
    "ACTIONS_ENABLE_DEL_NEURONS_01",
    "ACTIONS_ENABLE_DEL_NEURONS_05",
    "ACTIONS_ENABLE_DEL_NEURONS_09",
)

STARTERS = {
    "narrow": (4, 32),  # 33390 params
    "base": (8, 38),  # 79060 params
    "mid": (12, 45),  # 140389 params
    "deep": (16, 48),  # 199914 params
}

SIMULATION_ALGS = {
    "montecarlo": montecarlo_alg,
    "greedy": greedy_alg,
    "sequential_halving_beam": sequential_halving_beam_alg,
    "ugape_deepen": ugape_deepen_alg,
    "best_first": best_first_alg,
}

GROUPS = {
    "starter": ("narrow", "base", "mid", "deep"),
    "epochs": (5, 10, 20),
    "generations": (10, 20),
    "simulation_alg": tuple(SIMULATION_ALGS),
    "lr_schedule": ("composed_exponential", "composed_step", "composed_cosine"),
    "lr_alpha": (0.001, 0.01, 0.03),
    "simulation_time": (60.0, 120.0, 240.0),
    "simulation_epochs": (10, 15, 20),
    "simulation_set_size": (100, 500, 1000),
    "simulation_scheduler": ("always", "slope_2deg", "slope_3deg"),
}


def apply_residual_conv_pool_patch():
    """Force residual-into-linear skips to use average pool when growth adds them."""
    return patch.object(
        growingnn_config,
        "RES_CONV_TO_LINEAR_GLOBAL_POOL_TYPE",
        RESIDUAL_CONV_POOL_TYPE,
    )


_ALG_FOLDER = {
    "montecarlo": "mc",
    "greedy": "gr",
    "sequential_halving_beam": "shb",
    "ugape_deepen": "ugd",
    "best_first": "bf",
}
_LR_FOLDER = {
    "composed_exponential": "exp",
    "composed_step": "step",
    "composed_cosine": "cos",
}
_SCHED_FOLDER = {
    "always": "alw",
    "slope_2deg": "s2",
    "slope_3deg": "s3",
}


def _folder_num(value: object) -> str:
    number = float(value)
    return str(int(number)) if number.is_integer() else str(value)


def combo_folder_name(combo: dict[str, object]) -> str:
    """Build a compact folder name for one search combo."""
    return (
        f"st{combo['starter']}_g{_folder_num(combo['generations'])}_ep{_folder_num(combo['epochs'])}"
        f"_{_ALG_FOLDER[str(combo['simulation_alg'])]}_{_LR_FOLDER[str(combo['lr_schedule'])]}"
        f"_a{_folder_num(combo['lr_alpha'])}_t{_folder_num(combo['simulation_time'])}"
        f"_se{_folder_num(combo['simulation_epochs'])}_sz{_folder_num(combo['simulation_set_size'])}"
        f"_{_SCHED_FOLDER[str(combo['simulation_scheduler'])]}"
    )


def _lr_scheduler_for_schedule_id(schedule_id: str, hp: dict[str, object]):
    """Build a composed LR schedule with step size total_epochs // 3 for this combo."""
    total_epochs = int(hp["generations"]) * int(hp["epochs"])
    base_name = {
        "composed_cosine": "cosine",
        "composed_step": "step",
        "composed_exponential": "exponential",
    }.get(schedule_id)
    if base_name is None:
        raise ValueError(f"unknown lr_schedule {schedule_id!r}")
    return build_composed_learning_rate_scheduler(
        base_name,
        total_epochs=total_epochs,
        initial_lr=float(hp["lr_alpha"]),
        warmup_iterations=WARMUP_ITERATIONS,
        k=WARMUP_STEEPNESS,
        eta_min=ETA_MIN,
        step_size=max(1, total_epochs // 3),
        gamma=EXPONENTIAL_GAMMA if base_name == "exponential" else STEP_GAMMA,
    )


def hyperparameters_for_combo(combo: dict[str, object]) -> dict[str, object]:
    """Map one search combo onto the GrowingNN hyperparameter dict."""
    channels, hidden_dim = STARTERS[str(combo["starter"])]
    schedule_id = str(combo["lr_schedule"])
    return {
        "generations": int(combo["generations"]),
        "epochs": int(combo["epochs"]),
        "batch_size": BATCH_SIZE,
        "lr_alpha": float(combo["lr_alpha"]),
        "simulation_time": float(combo["simulation_time"]),
        "simulation_epochs": int(combo["simulation_epochs"]),
        "simulation_set_size": int(combo["simulation_set_size"]),
        "target_accuracy": TARGET_ACCURACY,
        "score_weight_acc": SCORE_WEIGHT_ACC,
        "score_weight_countw": SCORE_WEIGHT_COUNTW,
        "model_channels": channels,
        "model_hidden_dim": hidden_dim,
        "score_accuracy_metric": SCORE_ACCURACY_METRIC,
        "simulation_alg_id": str(combo["simulation_alg"]),
        "simulation_alg": SIMULATION_ALGS[str(combo["simulation_alg"])],
        "starter": str(combo["starter"]),
        "simulation_scheduler": str(combo["simulation_scheduler"]),
        "lr_scheduler_factory": (
            lambda hp, sid=schedule_id: _lr_scheduler_for_schedule_id(sid, hp)
        ),
    }


def _simulation_scheduler_for_combo(combo: dict[str, object], hp: dict[str, object]):
    time_s = float(hp["simulation_time"])
    sim_epochs = int(hp["simulation_epochs"])
    kind = str(combo["simulation_scheduler"])
    if kind == "always":
        return AlwaysSimulationScheduler(time_s, sim_epochs)
    if kind == "slope_2deg":
        return SlopeEstimationSimulationScheduler(time_s, sim_epochs, angle_threshold=2.0)
    if kind == "slope_3deg":
        return SlopeEstimationSimulationScheduler(time_s, sim_epochs, angle_threshold=3.0)
    raise ValueError(f"unknown simulation_scheduler {kind!r}")


def _running_config(hp: dict[str, object], combo: dict[str, object], device: torch.device, board) -> RunningConfig:
    cfg = common._running_config(hp, device, board)
    for flag in _NEURON_RESIZE_FLAGS:
        setattr(cfg, flag, False)
    cfg.simulation_scheduler = _simulation_scheduler_for_combo(combo, hp)
    return cfg


def create_model(hp: dict[str, object]) -> fx.GraphModule:
    """Step 1: build and FX-trace the sequential CIFAR starter."""
    return fx.symbolic_trace(train_cifar10._build_model(hp))


def train_model(
    model: fx.GraphModule,
    hp: dict[str, object],
    combo: dict[str, object],
    *,
    data: train_cifar10.Cifar10Data,
    device: torch.device,
    seed: int,
    board: ExperimentBoard | None,
):
    """Step 2: run GrowingNN train_generations on this combo."""
    config = _running_config(hp, combo, device, board)
    train_loader, val_loader, clean_train_loader = data.loaders(int(hp["batch_size"]))
    sim_train, sim_val = config.simulation_set_generator.generate(
        clean_train_loader,
        val_loader,
        config.simulation_set_size,
        seed=seed,
        model=model,
    )
    return train_generations(
        model,
        train_loader,
        val_loader,
        config,
        sim_train_loader=sim_train,
        sim_val_loader=sim_val,
    )


def save_run(run_dir: Path, summary: dict, trained_model, start_params: int) -> float:
    """Step 3: write history and plots. Return best val_acc."""
    common._save_artifacts(run_dir, HISTORY_FILENAME, summary)
    val_acc = float(max(summary["val_acc"]))
    params_after = GraphStructureQuery.get_amount_of_parameters(trained_model)
    logger.info("Saved %s val_acc=%.4f params %s -> %s", run_dir, val_acc, start_params, params_after)
    return val_acc


def evaluate_combo(
    combo: dict[str, object],
    *,
    data: train_cifar10.Cifar10Data,
    device: torch.device,
    seed: int,
    board_enabled: bool,
) -> tuple[float, float]:
    """Create model, train, save. CIFAR val is the official test split, so test_acc = val_acc."""
    hp = hyperparameters_for_combo(combo)
    folder = combo_folder_name(combo)
    run_dir = RUNS_DIR / folder / f"seed_{seed}"
    history_path = run_dir / HISTORY_FILENAME
    if history_path.is_file():
        history = torch.load(history_path, map_location="cpu", weights_only=True)
        val_acc = float(max(history["val_acc"]))
        logger.info("Reusing existing %s val_acc=%.4f", run_dir, val_acc)
        return val_acc, val_acc
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    model = create_model(hp)
    start_params = GraphStructureQuery.get_amount_of_parameters(model)
    board = None
    if board_enabled:
        board = ExperimentBoard(
            run_dir / "board",
            experiment_name=f"CIFAR-10 exp008 {folder} | seed {seed}",
            dataset="CIFAR-10",
            device=str(device),
        )
    trained, summary = train_model(
        model, hp, combo, data=data, device=device, seed=seed, board=board
    )
    val_acc = save_run(run_dir, summary, trained, start_params)
    return val_acc, val_acc


def parse_exp008_cli() -> argparse.Namespace:
    """Parse search budget and board output."""
    parser = argparse.ArgumentParser(
        description="Experiment 008: CIFAR-10 adaptive meta-parameter search"
    )
    parser.add_argument("--board", choices=("true", "false"), default="true")
    parser.add_argument("--max-iters", type=int, default=MAX_ITERS)
    parser.add_argument("--n-init", type=int, default=N_INIT)
    parser.add_argument("--tau", type=float, default=TAU)
    parser.add_argument("--beta", type=float, default=BETA)
    args = parser.parse_args()
    args.board = args.board == "true"
    return args


def print_groups() -> None:
    """Print searched groups and live output paths."""
    print("Exp 008 CIFAR-10 adaptive meta-parameter search groups:")
    for axis, values in GROUPS.items():
        print(f"  {axis}: {list(values)}")
    print(f"Live status: {RUNS_DIR / 'adaptive_search.md'}")
    print("Smoke: python experiments/train_cifar10_exp008_initial_package.py --max-iters 1")


if __name__ == "__main__":
    args = parse_exp008_cli()
    configure_deterministic_seeding()
    device = torch.device("cuda")
    common.require_cuda(device)
    data = train_cifar10.Cifar10Data(train_cifar10.DATA_DIR)
    data.prepare()
    print(f"Exp 008 write target: {RUNS_DIR}")
    print_groups()
    search = AdaptiveMetaParameterSearch(
        GROUPS,
        RUNS_DIR,
        max_iters=args.max_iters,
        n_init=args.n_init,
        tau=args.tau,
        beta=args.beta,
    )
    with apply_residual_conv_pool_patch():
        while True:
            combo = search.next_config()
            if combo is None:
                break
            seed = SEED_BASE + len(search.trials)
            print(f"trial={len(search.trials) + 1} seed={seed} combo={combo}")
            val_acc, test_acc = evaluate_combo(
                combo, data=data, device=device, seed=seed, board_enabled=args.board
            )
            search.record(combo, val_acc, test_acc)
            print(f"recorded val_acc={val_acc:.4f} best={search.best['val_acc']:.4f}")
    print(f"Search finished. Live files under {RUNS_DIR}")
