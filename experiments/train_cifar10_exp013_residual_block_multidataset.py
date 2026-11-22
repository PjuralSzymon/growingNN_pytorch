"""Experiment 013: archived residual-block comparison across four image datasets.

The active comparison is Experiment 012. This script remains so the interim
MNIST, CIFAR-10, CIFAR-100, and Tiny ImageNet runs can be resumed.

Matched enabled/disabled arms use the same explicit shape-preserving residual
starter, training recipe, and seeds. Training ends when validation accuracy
reaches 90%, or after the configured generation limit.

Default: MNIST, CIFAR-10, CIFAR-100, Tiny ImageNet; seeds 1, 2, 3;
30 generations x 10 epochs maximum. Tiny ImageNet is not full ImageNet.

    pipenv run python -u experiments/train_cifar10_exp013_residual_block_multidataset.py
    pipenv run python -u experiments/train_cifar10_exp013_residual_block_multidataset.py mnist cifar10
"""

from __future__ import annotations

import argparse
import json
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
from experiments.train_cifar10_exp009_best_reference import get_datasets
from growingnn.board.experiment_board import ExperimentBoard
from growingnn.core.config import RunningConfig
from growingnn.simulation.score_functions.simulation_score import SimulationScore
from growingnn.simulation.simulation_schedulers import AlwaysSimulationScheduler
from growingnn.simulation.simulation_set import sample_loaders
from growingnn.training.stoppers import AccuracyStopper
from growingnn.training.trainer import train_generations
from growingnn.utils.seed import seed_all

# Folder name stays exp012 so histories saved before the renumber are reused.
EXPERIMENT_ID = "exp012_residual_block_multidataset"
RUNS_DIR = ROOT / "experiments" / "output" / "train_cifar10" / "runs" / EXPERIMENT_ID
DATASETS = ("mnist", "cifar10", "cifar100", "tiny_imagenet")
ARMS = ("block_actions_on", "block_actions_off")
DEFAULT_SEEDS = (1, 2, 3)
DEFAULT_GENERATIONS = 30
DEFAULT_EPOCHS = 10
BATCH_SIZE = 128
TARGET_VALIDATION_ACCURACY = 0.90
SIMULATION_SECONDS = 8.0
SIMULATION_EPOCHS = 1
SIMULATION_SET_SIZE = 128


class ResidualStarter(nn.Module):
    """Small common stem, identity-shape residual block, pool, and classifier."""

    def __init__(self, input_channels: int, class_count: int) -> None:
        super().__init__()
        self.stem = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(16)
        self.block_conv = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False)
        self.block_bn = nn.BatchNorm2d(16)
        self.classifier = nn.Linear(16, class_count)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.stem_bn(self.stem(x)))
        residual = self.block_bn(self.block_conv(x))
        x = F.relu(x + residual)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.classifier(x)


def _dataset_shape(dataset: str) -> tuple[int, int]:
    if dataset == "mnist":
        return 1, 10
    if dataset == "cifar10":
        return 3, 10
    if dataset == "cifar100":
        return 3, 100
    if dataset == "tiny_imagenet":
        return 3, 200
    raise ValueError(f"unknown dataset: {dataset}")


def _claim_run_dir(run_dir: Path) -> bool:
    """Claim a run directory; preserve completed runs and archive failed attempts."""
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        run_dir.mkdir(exist_ok=False)
        return True
    except FileExistsError:
        if (run_dir / "history.pt").is_file():
            print(f"SKIP completed run: {run_dir}", flush=True)
            return False
        if not (run_dir / "FAILED.txt").is_file():
            print(f"SKIP existing run directory without failure marker: {run_dir}", flush=True)
            return False
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archived = run_dir.with_name(f"{run_dir.name}_failed_{stamp}")
        suffix = 1
        while archived.exists():
            archived = run_dir.with_name(f"{run_dir.name}_failed_{stamp}_{suffix}")
            suffix += 1
        run_dir.rename(archived)
        run_dir.mkdir()
        print(f"RETRY archived failed attempt: {archived}", flush=True)
        return True


def _train_one(data, dataset: str, arm: str, seed: int, generations: int, epochs: int) -> None:
    run_dir = RUNS_DIR / dataset / arm / f"seed_{seed}"
    if not _claim_run_dir(run_dir):
        return

    try:
        seed_all(seed)
        input_channels, class_count = _dataset_shape(dataset)
        model = ResidualStarter(input_channels, class_count)
        train_loader, val_loader, clean_loader = data.loaders(BATCH_SIZE)
        train_loader = with_cutout(train_loader)
        test_loader = data.test_loader(BATCH_SIZE)
        sim_train, sim_val = sample_loaders(clean_loader, val_loader, SIMULATION_SET_SIZE, seed=seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        board = ExperimentBoard(
            run_dir / "board",
            experiment_id=f"{EXPERIMENT_ID}_{dataset}_{arm}_seed{seed}",
            experiment_name="Experiment 013 residual-block action multi-dataset study",
            dataset=dataset.upper(),
            device=str(device),
        )
        enabled = arm == "block_actions_on"
        config = RunningConfig(
            generations=generations,
            epochs=epochs,
            lr_scheduler=CosineDecayScheduler(total_epochs=generations * epochs),
            stopper=AccuracyStopper(
                target_accuracy=TARGET_VALIDATION_ACCURACY,
                metric_name="validation accuracy",
                metric_key="val_acc",
            ),
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
        config.ACTIONS_ENABLE_RESIDUAL_BLOCK_GROW = enabled
        config.ACTIONS_ENABLE_RESIDUAL_BLOCK_SHRINK = enabled
        print(
            f"EXP013 dataset={dataset} arm={arm} seed={seed} device={device} "
            f"generations_max={generations} epochs_per_generation={epochs} "
            f"validation_stop={TARGET_VALIDATION_ACCURACY:.0%} residual_block_actions={enabled}",
            flush=True,
        )
        _, summary = train_generations(
            model,
            train_loader,
            val_loader,
            config,
            sim_train_loader=sim_train,
            sim_val_loader=sim_val,
            test_loader=test_loader,
        )
        final_val = float(summary["val_acc"][-1]) if summary["val_acc"] else 0.0
        history = {
            "experiment": EXPERIMENT_ID,
            "dataset": dataset,
            "arm": arm,
            "seed": seed,
            "starter": "ResidualStarter(conv16_bn_relu, conv16_bn_residual, global_avg_pool, linear_head)",
            "residual_block_actions_enabled": enabled,
            "weight_initialization": "quasi_identity",
            "target_validation_accuracy": TARGET_VALIDATION_ACCURACY,
            "stopped_by_validation_target": final_val >= TARGET_VALIDATION_ACCURACY,
            "final_validation_accuracy": final_val,
            "generations_max": generations,
            "generations_completed": len(summary["generation"]),
            "epochs_per_generation_max": epochs,
            "epochs_completed": len(summary["train_acc"]),
            "simulation_seconds": SIMULATION_SECONDS,
            "simulation_epochs": SIMULATION_EPOCHS,
            "simulation_set_size": SIMULATION_SET_SIZE,
            **{key: list(value) for key, value in summary.items()},
        }
        torch.save(history, run_dir / "history.pt")
        (run_dir / "metadata.json").write_text(json.dumps({k: v for k, v in history.items() if not isinstance(v, list)}, indent=2), encoding="utf-8")
        action_counts: dict[str, int] = {}
        for name in summary.get("action_name", []):
            action_counts[name] = action_counts.get(name, 0) + 1
        print(
            f"DONE {dataset}/{arm}/seed_{seed} generations={history['generations_completed']} "
            f"epochs={history['epochs_completed']} final_val={final_val:.4f} actions={action_counts}",
            flush=True,
        )
    except Exception as exc:
        (run_dir / "FAILED.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "datasets",
        nargs="*",
        metavar="{" + ",".join(DATASETS) + "}",
        help="datasets to run (defaults to all four)",
    )
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    args = parser.parse_args()
    if args.generations < 1 or args.epochs < 1:
        parser.error("--generations and --epochs must be positive")
    if not args.seeds:
        parser.error("provide at least one seed")
    selected_datasets = tuple(args.datasets) if args.datasets else DATASETS
    invalid_datasets = sorted(set(selected_datasets) - set(DATASETS))
    if invalid_datasets:
        parser.error(
            f"unknown dataset(s): {', '.join(invalid_datasets)}; "
            f"choose from {', '.join(DATASETS)}"
        )
    for dataset in selected_datasets:
        data = get_datasets(dataset)
        for arm in args.arms:
            for seed in args.seeds:
                _train_one(data, dataset, arm, seed, args.generations, args.epochs)


if __name__ == "__main__":
    main()
