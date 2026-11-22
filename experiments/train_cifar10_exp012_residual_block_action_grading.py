"""Experiment 012: grade residual-block actions on MNIST and CIFAR-10.

Both arms start from the same identity residual block. One arm can grow or
shrink residual blocks. The other cannot. Search uses the MCTS settings
selected by Experiment 010.

    pipenv run python -u experiments/train_cifar10_exp012_residual_block_action_grading.py
    pipenv run python -u experiments/train_cifar10_exp012_residual_block_action_grading.py mnist
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

import growingnn.core.config as project_config
import growingnn.simulation.simulation_algorithms.montecarlo_alg as montecarlo_alg
from experiments.train_cifar10_exp008_generation_only import CosineDecayScheduler, with_cutout
from experiments.train_cifar10_exp009_best_reference import get_datasets
from growingnn.board.experiment_board import ExperimentBoard
from growingnn.core.config import RunningConfig
from growingnn.simulation.score_functions.simulation_score import SimulationScore
from growingnn.simulation.simulation_schedulers import AlwaysSimulationScheduler
from growingnn.simulation.simulation_set import sample_loaders
from growingnn.training.lr_scheduler_action import ActionLearningRateScheduler, ScheduleMode
from growingnn.training.stoppers import AccuracyStopper
from growingnn.training.trainer import train_generations
from growingnn.utils.seed import seed_all

EXPERIMENT_ID = "exp012_residual_block_action_grading"
RUNS_DIR = ROOT / "experiments" / "output" / "train_cifar10" / "runs" / EXPERIMENT_ID
DATASETS = ("mnist", "cifar10")
ARMS = ("block_actions_on", "block_actions_off")
DEFAULT_SEEDS = (1, 2, 3)
GENERATIONS = 30
EPOCHS = 10
BATCH_SIZE = 128
TARGET_VALIDATION_ACCURACY = 0.90
SIMULATION_SECONDS = 32.0
SIMULATION_EPOCHS = 1
SIMULATION_SET_PERCENT = 10.0
# Pass-2 winners from Experiment 010, selected by mean final validation accuracy.
MCTS_SETTINGS = {
    "MCTS_UCB1_C": 1.0,
    "MCTS_UCB1_USE_SQRT": False,
    "MCTS_PROPAGATE_ROLLOUT_VALUE": True,
    "MCTS_ROLLOUT_DEPTH": 1,
    "MCTS_ROLLOUT_EPOCHS": 1,
    "MCTS_ROLLOUT_LR": 0.0001,
    "SIMULATION_MIN_ALGORITHM_ITERATION_RUNS": 3,
}


class ResidualStarter(nn.Module):
    """Small stem, one identity-shape residual block, pool, and classifier."""

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


def simulation_sample_size(train_size: int, percent: float = SIMULATION_SET_PERCENT) -> int:
    """Return the simulation-set size for a percentage of the clean training split."""
    if train_size < 1:
        raise ValueError(f"train_size must be positive, got {train_size}")
    if percent <= 0:
        raise ValueError(f"percent must be positive, got {percent}")
    return max(1, min(train_size, round(train_size * percent / 100.0)))


def apply_experiment_010_mcts_settings() -> dict:
    """Install the Experiment 010 pass-2 MCTS winners and return the previous globals."""
    previous = {name: getattr(project_config, name) for name in MCTS_SETTINGS}
    for name, value in MCTS_SETTINGS.items():
        if name == "MCTS_ROLLOUT_LR":
            value = ActionLearningRateScheduler(ScheduleMode.CONSTANT, alpha=float(value))
        setattr(project_config, name, value)
    return previous


def restore_mcts_settings(previous: dict) -> None:
    """Put project MCTS globals back to the values captured before a run."""
    for name, value in previous.items():
        setattr(project_config, name, value)


def _dataset_shape(dataset: str) -> tuple[int, int]:
    if dataset == "mnist":
        return 1, 10
    if dataset == "cifar10":
        return 3, 10
    raise ValueError(f"unknown dataset: {dataset}")


def _claim_run_dir(run_dir: Path) -> bool:
    """Claim a run directory, keep completed runs, and archive unfinished attempts."""
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        run_dir.mkdir(exist_ok=False)
        return True
    except FileExistsError:
        if (run_dir / "history.pt").is_file():
            print(f"SKIP completed run: {run_dir}", flush=True)
            return False
        label = "failed" if (run_dir / "FAILED.txt").is_file() else "incomplete"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archived = run_dir.with_name(f"{run_dir.name}_{label}_{stamp}")
        suffix = 1
        while archived.exists():
            archived = run_dir.with_name(f"{run_dir.name}_{label}_{stamp}_{suffix}")
            suffix += 1
        run_dir.rename(archived)
        run_dir.mkdir()
        print(f"RETRY archived {label} attempt: {archived}", flush=True)
        return True


def _train_one(data, dataset: str, arm: str, seed: int) -> None:
    run_dir = RUNS_DIR / dataset / arm / f"seed_{seed}"
    if not _claim_run_dir(run_dir):
        return

    try:
        apply_experiment_010_mcts_settings()
        seed_all(seed)
        input_channels, class_count = _dataset_shape(dataset)
        model = ResidualStarter(input_channels, class_count)
        train_loader, val_loader, clean_loader = data.loaders(BATCH_SIZE)
        train_loader = with_cutout(train_loader)
        test_loader = data.test_loader(BATCH_SIZE)
        sample_size = simulation_sample_size(len(clean_loader.dataset))
        sim_train, sim_val = sample_loaders(clean_loader, val_loader, sample_size, seed=seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        board = ExperimentBoard(
            run_dir / "board",
            experiment_id=f"{EXPERIMENT_ID}_{dataset}_{arm}_seed{seed}",
            experiment_name="Experiment 012 residual-block action grading",
            dataset=dataset.upper(),
            device=str(device),
        )
        enabled = arm == "block_actions_on"
        config = RunningConfig(
            generations=GENERATIONS,
            epochs=EPOCHS,
            lr_scheduler=CosineDecayScheduler(total_epochs=GENERATIONS * EPOCHS),
            stopper=AccuracyStopper(
                target_accuracy=TARGET_VALIDATION_ACCURACY,
                metric_name="validation accuracy",
                metric_key="val_acc",
            ),
            simulation_alg=montecarlo_alg,
            simulation_scheduler=AlwaysSimulationScheduler(SIMULATION_SECONDS, SIMULATION_EPOCHS),
            simulation_score=SimulationScore(weight_acc=1.0, weight_countW=0.0),
            simulation_set_size=len(sim_train.dataset),
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
            f"EXP012 dataset={dataset} arm={arm} seed={seed} device={device} "
            f"generations_max={GENERATIONS} epochs={EPOCHS} "
            f"validation_stop={TARGET_VALIDATION_ACCURACY:.0%} sim={SIMULATION_SECONDS}s "
            f"sim_train={len(sim_train.dataset)}/{len(clean_loader.dataset)} "
            f"residual_block_actions={enabled}",
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
            "generations_max": GENERATIONS,
            "generations_completed": len(summary["generation"]),
            "epochs_per_generation": EPOCHS,
            "simulation_seconds": SIMULATION_SECONDS,
            "simulation_epochs": SIMULATION_EPOCHS,
            "simulation_set_percent": SIMULATION_SET_PERCENT,
            "simulation_train_size": len(sim_train.dataset),
            "simulation_val_size": len(sim_val.dataset),
            "mcts_settings": dict(MCTS_SETTINGS),
            **{key: list(value) for key, value in summary.items()},
        }
        torch.save(history, run_dir / "history.pt")
        metadata = {key: value for key, value in history.items() if not isinstance(value, list)}
        (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        action_counts: dict[str, int] = {}
        for name in summary.get("action_name", []):
            action_counts[name] = action_counts.get(name, 0) + 1
        print(f"DONE {dataset}/{arm}/seed_{seed} actions={action_counts} board={run_dir / 'board'}", flush=True)
    except Exception as exc:
        (run_dir / "FAILED.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", choices=DATASETS)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    args = parser.parse_args()
    if not args.seeds:
        parser.error("provide at least one seed")
    previous = apply_experiment_010_mcts_settings()
    try:
        for dataset in args.datasets or DATASETS:
            data = get_datasets(dataset)
            for arm in args.arms:
                for seed in args.seeds:
                    _train_one(data, dataset, arm, seed)
    finally:
        restore_mcts_settings(previous)


if __name__ == "__main__":
    main()
