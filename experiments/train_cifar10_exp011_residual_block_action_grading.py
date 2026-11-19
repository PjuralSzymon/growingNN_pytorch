"""Experiment 011: grade residual-block actions on CIFAR-10 and CIFAR-100.

The starter exposes a shape-preserving identity residual block. The current
runner always uses quasi-identity initialization for duplicated blocks;
historical Experiment 011 artifacts preserve the earlier initializer comparison.

    python .\experiments\train_cifar10_exp011_residual_block_action_grading.py
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import growingnn.simulation.simulation_algorithms.montecarlo_alg as montecarlo_alg
from experiments.train_cifar10_exp008_generation_only import CosineDecayScheduler, with_cutout
from experiments.train_cifar10_exp009_best_reference import DataFactory
from growingnn.board.experiment_board import ExperimentBoard
from growingnn.core.config import RunningConfig
from growingnn.simulation.score_functions.simulation_score import SimulationScore
from growingnn.simulation.simulation_schedulers import AlwaysSimulationScheduler
from growingnn.simulation.simulation_set import sample_loaders
from growingnn.training.stoppers import AccuracyStopper
from growingnn.training.trainer import train_generations
from growingnn.utils.seed import seed_all

RUNS_DIR = ROOT / "experiments" / "output" / "train_cifar10" / "runs" / "exp011_residual_block_action_grading"
DATASETS = ("cifar10", "cifar100")
SEEDS = (1, 2, 3)
ARMS = ("block_actions_on", "block_actions_off")
GENERATIONS = 10
EPOCHS = 5
BATCH_SIZE = 128
SIMULATION_SECONDS = 4.0
SIMULATION_EPOCHS = 1
SIMULATION_SET_SIZE = 256


class ResidualCifarStarter(nn.Module):
    """Small CIFAR-10 classifier with one explicit identity residual block."""

    def __init__(self, classes: int = 10) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(16)
        self.block_conv = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False)
        self.block_bn = nn.BatchNorm2d(16)
        self.head = nn.Linear(16, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.stem_bn(self.stem(x)))
        residual = self.block_bn(self.block_conv(x))
        x = F.relu(x + residual)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.head(x)


def _dataset_root(dataset: str) -> Path:
    # Preserve completed baseline paths; CIFAR-100 is nested below the same root.
    return RUNS_DIR if dataset == "cifar10" else RUNS_DIR / dataset


def _run_root(dataset: str) -> Path:
    return _dataset_root(dataset) / "weight_init_quasi_identity"


def _train_one(data, dataset: str, arm: str, seed: int) -> None:
    initialization = "quasi_identity"
    run_dir = _run_root(dataset) / arm / f"seed_{seed}"
    history_path = run_dir / "history.pt"
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        run_dir.mkdir(exist_ok=False)
    except FileExistsError:
        if history_path.is_file():
            print(f"SKIP {dataset}/{arm}/seed_{seed}: history exists at {run_dir}", flush=True)
            return
        failure_marker = run_dir / "FAILED.txt"
        if not failure_marker.is_file():
            print(f"SKIP {dataset}/{arm}/seed_{seed}: run folder exists without a failure marker at {run_dir}", flush=True)
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archived = run_dir.with_name(f"{run_dir.name}_failed_{stamp}")
        suffix = 1
        while archived.exists():
            archived = run_dir.with_name(f"{run_dir.name}_failed_{stamp}_{suffix}")
            suffix += 1
        run_dir.rename(archived)
        print(f"RETRY {dataset}/{arm}/seed_{seed}: archived failed attempt at {archived}", flush=True)
        run_dir.mkdir()

    seed_all(seed)
    classes = 10 if dataset == "cifar10" else 100
    model = ResidualCifarStarter(classes=classes)
    train_loader, val_loader, clean_loader = data.loaders(BATCH_SIZE)
    train_loader = with_cutout(train_loader)
    test_loader = data.test_loader(BATCH_SIZE)
    sim_train, sim_val = sample_loaders(clean_loader, val_loader, SIMULATION_SET_SIZE, seed=seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    board = ExperimentBoard(
        run_dir / "board",
        experiment_id=f"exp011_{initialization}_{arm}_seed{seed}",
        experiment_name=f"Experiment 011 residual-block initialization: {initialization}",
        dataset=dataset.upper(),
        device=str(device),
    )
    config = RunningConfig(
        generations=GENERATIONS,
        epochs=EPOCHS,
        lr_scheduler=CosineDecayScheduler(total_epochs=GENERATIONS * EPOCHS),
        stopper=AccuracyStopper(target_accuracy=0.999),
        simulation_alg=montecarlo_alg,
        simulation_scheduler=AlwaysSimulationScheduler(SIMULATION_SECONDS, SIMULATION_EPOCHS),
        simulation_score=SimulationScore(weight_acc=1.0, weight_countW=0.0),
        simulation_set_size=SIMULATION_SET_SIZE,
        criterion=nn.CrossEntropyLoss(label_smoothing=0.1),
        enable_experiment_board=True,
        experiment_board=board,
        device=device,
        momentum=0.9,
        weight_decay=0.0001,
        max_grad_norm=1.0,
        seq_conv_insert_before_global_pool=True,
    )
    enabled = arm == "block_actions_on"
    config.ACTIONS_ENABLE_RESIDUAL_BLOCK_GROW = enabled
    config.ACTIONS_ENABLE_RESIDUAL_BLOCK_SHRINK = enabled
    print(
        f"EXP011 dataset={dataset} arm={arm} seed={seed} init={initialization} device={device} initial_block=identity "
        f"generations={GENERATIONS} epochs={EPOCHS} sim={SIMULATION_SECONDS}s "
        f"board=on residual_block_actions={enabled}",
        flush=True,
    )
    try:
        _, summary = train_generations(
            model,
            train_loader,
            val_loader,
            config,
            sim_train_loader=sim_train,
            sim_val_loader=sim_val,
            test_loader=test_loader,
        )
    except Exception as exc:
        (run_dir / "FAILED.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        raise
    payload = {
        "experiment": "011_residual_block_action_grading",
        "dataset": dataset,
        "arm": arm,
        "duplicate_initialization": initialization,
        "seed": seed,
        "starter": f"ResidualCifarStarter(conv16, identity residual conv16, global pool, linear{classes})",
        "residual_block_actions_enabled": enabled,
        "generations": GENERATIONS,
        "epochs_per_generation": EPOCHS,
        "simulation_seconds": SIMULATION_SECONDS,
        "simulation_epochs": SIMULATION_EPOCHS,
        "simulation_set_size": SIMULATION_SET_SIZE,
        **{key: list(value) for key, value in summary.items()},
    }
    torch.save(payload, history_path)
    counts: dict[str, int] = {}
    for name in summary.get("action_name", []):
        counts[name] = counts.get(name, 0) + 1
    print(f"DONE {dataset}/{arm}/seed_{seed} actions={counts} board={run_dir / 'board'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", choices=DATASETS)
    parser.add_argument(
        "--arms", nargs="+", choices=ARMS, default=("block_actions_on",),
        help="run residual-block actions enabled or disabled",
    )
    args = parser.parse_args()
    selected_datasets = args.datasets or DATASETS
    for dataset in selected_datasets:
        data = DataFactory.getcifar10() if dataset == "cifar10" else DataFactory.getcifar100()
        for arm in args.arms:
            for seed in SEEDS:
                _train_one(data, dataset, arm, seed)


if __name__ == "__main__":
    main()
