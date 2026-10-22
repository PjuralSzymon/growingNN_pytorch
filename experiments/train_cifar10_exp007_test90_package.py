"""
Experiment 007 driver — CIFAR-10 original growth versus the package that reached 90% test.

Two groups, seeds 110, 111, and 112. The official 10,000-image test set is logged.
Search uses the 5,000-image validation slice of the training set.

original
    Plain MinimalCifarNet. No BatchNorm. Sequential convs stay after the global pool.
    No BatchNorm-ReLU on a new conv. No gradient clip. Cosine learning rate 0.01.
    Slope gate 3 degrees. plain_growth. 8 generations x 10 epochs. No finish stage.

added
    FourLayerCifarNet with BatchNorm. New sequential convs go before the global pool
    and are followed by BatchNorm and ReLU. Gradient clip 1.0. Learning rate 0.1
    during 7 generations x 8 epochs. Only AddSeqConvLayer. Then a finish stage:
    step decay at epochs 30 and 60, label smoothing 0.1, 24 chunks of 5 epochs.
    Passing 0.90 official test accuracy does not stop the finish stage.

Charts and the result snapshot are written only after this script finishes.
Report:
documentation/website/content/experiments/experiment-007-cifar10-test90-package.md

Rerun only the added group, for the three seeds:

    pipenv run python experiments/train_cifar10_exp007_test90_package.py --group added

Each added seed writes history.pt. train_acc and val_acc are one value per epoch
through growth and the finish stage. test_acc is the official test at the end of
each growth generation and each finish chunk. test_epoch is the epoch of that measurement.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import growingnn.simulation.simulation_algorithms.montecarlo_alg as montecarlo_alg
from experiments import experiments_common as common
from experiments.train_cifar10 import (
    DATA_DIR,
    Cifar10Data,
    MinimalCifarNet,
    learning_rate_scheduler_factory,
    simulation_scheduler_factory,
)
from growingnn.core.config import RunningConfig
from growingnn.simulation.score_functions.simulation_score import SimulationScore
from growingnn.simulation.simulation_set import sample_loaders
from growingnn.training.stoppers import AccuracyStopper
from growingnn.training.trainer import train_generations
from growingnn.utils.seed import seed_all

RUNS_DIR = Path(__file__).resolve().parent / "output" / "train_cifar10" / "runs" / "exp007_test90_package"
SEEDS = (110, 111, 112)
GROUP_IDS = ("original", "added")

_ADDED_OFF_FLAGS = (
    "ACTIONS_ENABLE_DEL_LAYER",
    "ACTIONS_ENABLE_DEL_NEURONS_01",
    "ACTIONS_ENABLE_DEL_NEURONS_05",
    "ACTIONS_ENABLE_DEL_NEURONS_09",
    "ACTIONS_ENABLE_ADD_NEURONS_11",
    "ACTIONS_ENABLE_ADD_NEURONS_15",
    "ACTIONS_ENABLE_ADD_NEURONS_20",
    "ACTIONS_ENABLE_ADD_SEQ_DROPOUT_01",
    "ACTIONS_ENABLE_ADD_SEQ_DROPOUT_02",
    "ACTIONS_ENABLE_ADD_SEQ_DROPOUT_05",
    "ACTIONS_ENABLE_ADD_SEQ_LAYER",
    "ACTIONS_ENABLE_ADD_RES_LAYER",
    "ACTIONS_ENABLE_ADD_RES_CONV_LAYER",
)

ADDED_FINISH_DECAY_AT_01 = 30
ADDED_FINISH_DECAY_AT_001 = 60
ADDED_FINISH_CHUNKS = 24
ADDED_FINISH_EPOCHS_PER_CHUNK = 5
ADDED_LABEL_SMOOTHING = 0.1
_EPOCH_KEYS = ("train_loss", "train_acc", "val_loss", "val_acc", "lr", "param_count")
_ACTION_KEYS = (
    "action_generation",
    "action_name",
    "action_val_acc_before",
    "action_val_acc_after",
)


class StepDecayScheduler:
    """Hold 0.1, then 0.01, then 0.001. A structure change does not reset the epoch counter."""

    def __init__(self, at_01: int, at_001: int) -> None:
        self.global_epoch = 0
        self.at_01 = at_01
        self.at_001 = at_001

    def alpha_scheduler(self, _i: int, _iterations: int) -> float:
        epoch = self.global_epoch
        self.global_epoch += 1
        if epoch >= self.at_001:
            return 0.001
        if epoch >= self.at_01:
            return 0.01
        return 0.1

    def structure_changed(self) -> None:
        return None

    def learning_rate_config_board_labels(self) -> tuple[str, float]:
        return "StepDecay", 0.1


class FourLayerCifarNet(nn.Module):
    """Two convolutions with BatchNorm and a two-layer head. Search adds later convs."""

    def __init__(self, channels: int = 64, hidden_dim: int = 128, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.linear = nn.Linear(channels, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, num_classes, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.adaptive_avg_pool2d(x, 1)
        x = F.relu(self.linear(x.flatten(1)))
        return self.linear2(x)


def original_hyperparameters() -> dict[str, object]:
    """Product CIFAR recipe from train_cifar10.py before the 90% package."""
    return {
        "generations": 8,
        "epochs": 10,
        "batch_size": 128,
        "lr_alpha": 0.01,
        "simulation_time": 90.0,
        "simulation_epochs": 8,
        "simulation_set_size": 2000,
        "target_accuracy": 0.99,
        "score_weight_acc": 1.0,
        "score_weight_countw": 0.0,
        "model_channels": 32,
        "model_hidden_dim": 64,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "scheduler": "slope3",
        "lr_mode": "cosine",
        "plain_growth": True,
        "seq_conv_bn_relu": False,
        "seq_conv_before_global_pool": False,
        "max_grad_norm": None,
        "group_id": "original",
    }


def added_hyperparameters() -> dict[str, object]:
    """Settings of the seed-110 run that reached 90.01% official test accuracy."""
    return {
        "generations": 7,
        "epochs": 8,
        "batch_size": 128,
        "lr_alpha": 0.1,
        "simulation_time": 8.0,
        "simulation_epochs": 1,
        "simulation_set_size": 128,
        "target_accuracy": 0.99,
        "score_weight_acc": 1.0,
        "score_weight_countw": 0.0,
        "model_channels": 64,
        "model_hidden_dim": 128,
        "momentum": 0.9,
        "weight_decay": 0.0001,
        "scheduler": "always",
        "lr_mode": "step",
        "plain_growth": False,
        "seq_conv_bn_relu": True,
        "seq_conv_before_global_pool": True,
        "max_grad_norm": 1.0,
        "group_id": "added",
    }


def hyperparameters_for_group(group_id: str) -> dict[str, object]:
    """Return the hyperparameter dict for one Experiment 007 group."""
    if group_id == "original":
        return original_hyperparameters()
    if group_id == "added":
        return added_hyperparameters()
    raise ValueError(f"unknown group {group_id}")


def build_model(group_id: str, hp: dict[str, object]) -> nn.Module:
    """Build the starter module for one group."""
    channels = int(hp["model_channels"])
    hidden_dim = int(hp["model_hidden_dim"])
    if group_id == "original":
        return MinimalCifarNet(channels=channels, hidden_dim=hidden_dim)
    if group_id == "added":
        return FourLayerCifarNet(channels=channels, hidden_dim=hidden_dim)
    raise ValueError(f"unknown group {group_id}")


def running_config_for_group(group_id: str, device: torch.device | None = None) -> RunningConfig:
    """Build the RunningConfig that distinguishes the original recipe from the 90% package."""
    hp = hyperparameters_for_group(group_id)
    full = dict(hp)
    if group_id == "original":
        full["lr_scheduler_factory"] = learning_rate_scheduler_factory
        full["simulation_scheduler_factory"] = simulation_scheduler_factory
    else:
        full["lr_scheduler_factory"] = lambda _hp: StepDecayScheduler(at_01=1000, at_001=1000)
        full["simulation_scheduler_factory"] = simulation_scheduler_factory
    cfg = common._running_config(full, device or torch.device("cpu"), None)
    if group_id == "added":
        for flag in _ADDED_OFF_FLAGS:
            setattr(cfg, flag, False)
    return cfg


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Select the original group, the added group, or both, and an optional seed subset."""
    parser = argparse.ArgumentParser(description="Experiment 007: CIFAR-10 original growth versus the added package.")
    parser.add_argument("--group", choices=("original", "added", "both"), default="both")
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    args = parser.parse_args(argv)
    seeds = tuple(args.seeds) if args.seeds else SEEDS
    unknown = [seed for seed in seeds if seed not in SEEDS]
    if unknown:
        raise ValueError(f"seeds must be among {list(SEEDS)}: {unknown}")
    args.seeds = seeds
    return args


def added_history_from_growth(summary: dict, seed: int, epochs_per_generation: int) -> dict[str, object]:
    """Copy one added growth summary and mark each official test at its generation-end epoch."""
    n_test = len(summary["test_acc"])
    growth_epochs = len(summary["train_acc"])
    history: dict[str, object] = {
        "seed": seed,
        "growth_epochs": growth_epochs,
        "test_acc": [float(value) for value in summary["test_acc"]],
        "test_loss": [float(value) for value in summary.get("test_loss", [])],
        "test_epoch": [],
        "best_finish_test_acc": 0.0,
        "best_chunk": None,
        "best_epoch": None,
    }
    if n_test and growth_epochs == n_test * epochs_per_generation:
        history["test_epoch"] = [(index + 1) * epochs_per_generation for index in range(n_test)]
    for key in _EPOCH_KEYS:
        history[key] = [float(value) for value in summary[key]]
    for key in _ACTION_KEYS:
        history[key] = list(summary.get(key, []))
    return history


def append_finish_chunk(history: dict[str, object], chunk_history: dict, chunk: int, test_acc: float, test_loss: float) -> None:
    """Append one finish chunk's per-epoch train and validation, plus that chunk's official test."""
    n_epochs = len(chunk_history["train_acc"])
    for key in ("train_loss", "train_acc", "val_loss", "val_acc", "lr"):
        if key in chunk_history:
            history[key].extend(float(value) for value in chunk_history[key])
    if history.get("param_count"):
        history["param_count"].extend([history["param_count"][-1]] * n_epochs)
    history["test_acc"].append(float(test_acc))
    history["test_loss"].append(float(test_loss))
    epoch = int(history["growth_epochs"]) + ADDED_FINISH_EPOCHS_PER_CHUNK * (chunk + 1)
    history["test_epoch"].append(epoch)
    if float(test_acc) > float(history["best_finish_test_acc"]):
        history["best_finish_test_acc"] = float(test_acc)
        history["best_chunk"] = chunk
        history["best_epoch"] = epoch


def finish_added_training(
    model,
    train_loader,
    val_loader,
    test_loader,
    out_dir: Path,
    seed: int,
    history: dict[str, object] | None = None,
) -> float:
    """Train the grown graph for the full finish budget, including after official test accuracy passes 0.90."""
    from growingnn.training.gradient_descent import _evaluate, gradient_descent

    scheduler = StepDecayScheduler(ADDED_FINISH_DECAY_AT_01, ADDED_FINISH_DECAY_AT_001)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=ADDED_LABEL_SMOOTHING)
    if history is None:
        history = {
            "seed": seed,
            "growth_epochs": 0,
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "lr": [],
            "test_acc": [],
            "test_loss": [],
            "test_epoch": [],
            "best_finish_test_acc": 0.0,
            "best_chunk": None,
            "best_epoch": None,
        }
    best = 0.0
    passed_90 = False
    out_dir.mkdir(parents=True, exist_ok=True)
    for chunk in range(ADDED_FINISH_CHUNKS):
        model, chunk_history = gradient_descent(
            model, ADDED_FINISH_EPOCHS_PER_CHUNK, train_loader, val_loader, criterion, scheduler,
            optimizer=optimizer, quiet=False, print_every=1,
            momentum=0.9, weight_decay=1e-4, max_grad_norm=1.0, device="cpu",
        )
        test_loss, test_acc = _evaluate(model, test_loader, nn.CrossEntropyLoss(), torch.device("cpu"))
        append_finish_chunk(history, chunk_history, chunk, test_acc, test_loss)
        print(
            f"FINISH seed {seed} chunk {chunk} test_acc {test_acc:.4f} "
            f"train {chunk_history['train_acc'][-1]:.4f} val {chunk_history['val_acc'][-1]:.4f}",
            flush=True,
        )
        torch.save(history, out_dir / "history.pt")
        torch.save({"state_dict": model.state_dict(), "test_acc": test_acc, "seed": seed}, out_dir / "added_finish.pt")
        if test_acc > best:
            best = test_acc
            torch.save(
                {"state_dict": model.state_dict(), "test_acc": test_acc, "seed": seed, "chunk": chunk},
                out_dir / "added_finish_best.pt",
            )
        if test_acc >= 0.90 and not passed_90:
            passed_90 = True
            print(f"PASSED 0.90 seed {seed} chunk {chunk} test_acc {test_acc:.4f}", flush=True)
    print(f"FINISH seed {seed} done best_test_acc {best:.4f}", flush=True)
    return best


def run_added(data: Cifar10Data, out_dir: Path, seed: int) -> None:
    """Grow for six actions, then run the finish stage. history.pt keeps train, validation, and official test."""
    hp = added_hyperparameters()
    seed_all(seed)
    torch.set_num_threads(4)
    model = build_model("added", hp)
    train_loader, val_loader, clean_loader = data.loaders(int(hp["batch_size"]))
    test_loader = data.test_loader(int(hp["batch_size"]))
    sim_train, sim_val = sample_loaders(clean_loader, val_loader, int(hp["simulation_set_size"]), seed=seed)
    cfg = running_config_for_group("added", torch.device("cpu"))
    cfg.simulation_alg = montecarlo_alg
    cfg.simulation_score = SimulationScore(weight_acc=1.0, weight_countW=0.0)
    cfg.stopper = AccuracyStopper(target_accuracy=0.99)
    cfg.criterion = nn.CrossEntropyLoss()
    grown, summary = train_generations(
        model, train_loader, val_loader, cfg,
        sim_train_loader=sim_train, sim_val_loader=sim_val, test_loader=test_loader,
    )
    print(f"TEST seed {seed}", [round(value, 4) for value in summary["test_acc"]], flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = added_history_from_growth(summary, seed, int(hp["epochs"]))
    torch.save({"state_dict": grown.state_dict(), "test_acc": summary["test_acc"], "seed": seed}, out_dir / "added_growth.pt")
    torch.save(history, out_dir / "history.pt")
    print(f"HISTORY seed {seed} growth epochs {history['growth_epochs']} -> {out_dir / 'history.pt'}", flush=True)
    finish_added_training(grown, train_loader, val_loader, test_loader, out_dir, seed, history)


def run_original(data: Cifar10Data, seeds: tuple[int, ...] = SEEDS) -> None:
    """Run the original product recipe for the requested Experiment 007 seeds."""
    hp = original_hyperparameters()
    hp["lr_scheduler_factory"] = learning_rate_scheduler_factory
    hp["simulation_scheduler_factory"] = simulation_scheduler_factory
    definition = common.ExperimentDefinition(
        name="CIFAR-10 exp007 original",
        runs_dir=RUNS_DIR / "original",
        history_filename="history.pt",
        seeds=list(seeds),
        folder_name=lambda _hp: "original",
        model_factory=lambda item: build_model("original", item),
        loader_factory=lambda item: data.loaders(int(item["batch_size"])),
        test_loader_factory=lambda item: data.test_loader(int(item["batch_size"])),
        board_metadata=lambda _hp, folder, seed: (
            f"CIFAR-10 exp007 original | {folder} | seed {seed}",
            "CIFAR-10",
        ),
        save_fx_graphs=False,
    )
    common.run_experiment_grid(definition, [hp], device=torch.device("cpu"), board=False)


def main(argv: list[str] | None = None) -> None:
    """Run the selected Experiment 007 group."""
    args = parse_args(argv)
    print(f"Exp 007 group: {args.group}. Seeds {', '.join(str(seed) for seed in args.seeds)}.", flush=True)
    print(f"Write target: {RUNS_DIR}", flush=True)
    data = Cifar10Data(DATA_DIR)
    data.prepare()
    if args.group in ("original", "both"):
        run_original(data, args.seeds)
    if args.group in ("added", "both"):
        for seed in args.seeds:
            run_added(data, RUNS_DIR / "added" / f"seed_{seed}", seed)


if __name__ == "__main__":
    main()
