"""Experiment 010: test residual-block grow/shrink actions on three datasets.

Every run starts from the same topology: one small convolution, global average
pool, and a dataset-sized linear classifier. The only arm difference is
whether residual-block growth and shrink actions are enabled. All other action
types and training/search settings are held fixed. Two seeds are run per arm
and dataset; history.pt records train/validation/test histories and each
chosen action with generation and validation change.

    pipenv run python -u experiments/train_cifar10_exp010_residual_block_actions.py
    pipenv run python -u experiments/train_cifar10_exp010_residual_block_actions.py cifar10
"""

from __future__ import annotations

import sys
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
from growingnn.core.config import RunningConfig
from growingnn.simulation.score_functions.simulation_score import SimulationScore
from growingnn.simulation.simulation_schedulers import AlwaysSimulationScheduler
from growingnn.simulation.simulation_set import sample_loaders
from growingnn.training.stoppers import AccuracyStopper
from growingnn.training.trainer import train_generations
from growingnn.utils.seed import seed_all

RUNS_DIR = ROOT / "experiments" / "output" / "train_cifar10" / "runs" / "exp010_residual_block_actions"
DATASETS = ("mnist", "cifar10", "cifar100")
SEEDS = (1, 2)
ARMS = ("block_actions_on", "block_actions_off")
GENERATIONS = 20
EPOCHS = 10
BATCH_SIZE = 128
SIMULATION_SECONDS = 8.0
SIMULATION_EPOCHS = 1
SIMULATION_SET_SIZE = 128


class SmallStarter(nn.Module):
    """Same compact topology for every dataset; only channels/classes vary."""

    def __init__(self, input_channels: int, class_count: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(input_channels, 8, kernel_size=3, padding=1, bias=False)
        self.classifier = nn.Linear(8, class_count, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv(x))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.classifier(x)


def _model_for(dataset: str) -> SmallStarter:
    if dataset == "mnist":
        return SmallStarter(1, 10)
    if dataset == "cifar10":
        return SmallStarter(3, 10)
    if dataset == "cifar100":
        return SmallStarter(3, 100)
    raise ValueError(f"Unknown dataset: {dataset}")


def _train_one(data, dataset: str, arm: str, seed: int) -> None:
    output_dir = RUNS_DIR / dataset / arm / f"seed_{seed}"
    history_path = output_dir / "history.pt"
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_dir.mkdir(exist_ok=False)
    except FileExistsError:
        state = "history exists" if history_path.is_file() else "already started"
        print(f"SKIP {dataset}/{arm}/seed_{seed}: {state} at {output_dir}", flush=True)
        return

    seed_all(seed)
    model = _model_for(dataset)
    train_loader, val_loader, clean_loader = data.loaders(BATCH_SIZE)
    train_loader = with_cutout(train_loader)
    test_loader = data.test_loader(BATCH_SIZE)
    sim_train, sim_val = sample_loaders(clean_loader, val_loader, SIMULATION_SET_SIZE, seed=seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        f"EXP010 dataset={dataset} arm={arm} seed={seed} device={device} "
        f"starter=SmallStarter generations={GENERATIONS} epochs={EPOCHS} "
        f"residual_block_actions={enabled} all_other_actions=on",
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
    payload = {
        "experiment": "010_residual_block_actions",
        "dataset": dataset,
        "arm": arm,
        "seed": seed,
        "starter": "SmallStarter(conv8, adaptive_avg_pool, linear_head)",
        "residual_block_actions_enabled": enabled,
        "generations": GENERATIONS,
        "epochs_per_generation": EPOCHS,
        "simulation_seconds": SIMULATION_SECONDS,
        "simulation_epochs": SIMULATION_EPOCHS,
        "simulation_set_size": SIMULATION_SET_SIZE,
        **{key: list(value) for key, value in summary.items()},
    }
    torch.save(payload, history_path)
    action_counts: dict[str, int] = {}
    for action_name in summary.get("action_name", []):
        action_counts[action_name] = action_counts.get(action_name, 0) + 1
    print(f"DONE {dataset}/{arm}/seed_{seed}: actions={action_counts}", flush=True)


def main() -> None:
    selected = tuple(sys.argv[1:]) or DATASETS
    unknown = sorted(set(selected) - set(DATASETS))
    if unknown:
        raise SystemExit(f"Unknown dataset(s): {', '.join(unknown)}. Choose from {', '.join(DATASETS)}")
    for dataset in selected:
        data = get_datasets(dataset)
        for arm in ARMS:
            for seed in SEEDS:
                _train_one(data, dataset, arm, seed)


if __name__ == "__main__":
    main()
