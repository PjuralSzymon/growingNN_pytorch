"""
Experiment 008 driver — CIFAR-10 generation loop only.

Experiment 007 is unchanged. It still compares the original recipe with the
added package and its finish-chunk stage.

This experiment does not run that finish stage. Training is train_generations:
twelve generations of ten epochs, and one sequential convolution after every
generation except the last. The learning rate is a cosine from 0.1 to 0.001
across those epochs. The starter has 128 channels. Cutout is on the training
loader only. Seeds are 110, 111, and 112.

The official 10,000-image test set is logged. Search uses the 5,000-image
validation slice of the training set.

    pipenv run python -u experiments/train_cifar10_exp008_generation_only.py
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import growingnn.simulation.simulation_algorithms.montecarlo_alg as montecarlo_alg
from experiments.train_cifar10 import DATA_DIR, Cifar10Data
from experiments.train_cifar10_exp007_test90_package import (
    added_hyperparameters,
    build_model,
    running_config_for_group,
)
from growingnn.simulation.score_functions.simulation_score import SimulationScore
from growingnn.simulation.simulation_set import sample_loaders
from growingnn.training.stoppers import AccuracyStopper
from growingnn.training.trainer import train_generations
from growingnn.utils.seed import seed_all

RUNS_DIR = Path(__file__).resolve().parent / "output" / "train_cifar10" / "runs" / "exp008_generation_only"
SEEDS = (110, 111, 112)
GENERATIONS = 12
EPOCHS = 10
CHANNELS = 128
HIDDEN = 256
LABEL_SMOOTHING = 0.1
_EPOCH_KEYS = ("train_loss", "train_acc", "val_loss", "val_acc", "lr", "param_count")
_ACTION_KEYS = (
    "action_generation",
    "action_name",
    "action_val_acc_before",
    "action_val_acc_after",
)


class CosineDecayScheduler:
    """Cosine from 0.1 down to a floor. A structure change does not reset the epoch counter."""

    def __init__(self, total_epochs: int, floor: float = 0.001) -> None:
        self.global_epoch = 0
        self.total_epochs = max(int(total_epochs), 1)
        self.floor = float(floor)
        self._global_schedule_progress_frozen = False

    def alpha_scheduler(self, _i: int, _iterations: int) -> float:
        epoch = min(self.global_epoch, self.total_epochs - 1)
        if not self._global_schedule_progress_frozen:
            self.global_epoch += 1
        progress = epoch / max(self.total_epochs - 1, 1)
        return self.floor + 0.5 * (0.1 - self.floor) * (1.0 + math.cos(math.pi * progress))

    def freeze_global_schedule_progress(self):
        """Keep simulation scoring from advancing the training epoch counter."""
        from contextlib import contextmanager

        scheduler = self

        @contextmanager
        def _freeze():
            previous = scheduler._global_schedule_progress_frozen
            scheduler._global_schedule_progress_frozen = True
            try:
                yield scheduler
            finally:
                scheduler._global_schedule_progress_frozen = previous

        return _freeze()

    def structure_changed(self) -> None:
        return None

    def learning_rate_config_board_labels(self) -> tuple[str, float]:
        return "CosineDecay", 0.1


def with_cutout(loader: DataLoader) -> DataLoader:
    """Apply RandomErasing on the augmented training batches only."""
    base = loader.dataset
    erasing = transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0)

    class _Cutout(Dataset):
        def __len__(self) -> int:
            return len(base)

        def __getitem__(self, index: int):
            image, target = base[index]
            return erasing(image), target

    return DataLoader(_Cutout(), batch_size=loader.batch_size, shuffle=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Select an optional seed subset. There is no finish-chunk stage and no original group."""
    parser = argparse.ArgumentParser(description="Experiment 008: CIFAR-10 generation loop only.")
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    args = parser.parse_args(argv)
    seeds = tuple(args.seeds) if args.seeds else SEEDS
    unknown = [seed for seed in seeds if seed not in SEEDS]
    if unknown:
        raise ValueError(f"seeds must be among {list(SEEDS)}: {unknown}")
    args.seeds = seeds
    return args


def history_from_generations(summary: dict, seed: int, epochs_per_generation: int) -> dict[str, object]:
    """Copy one generation-loop summary. Official test is marked at each generation-end epoch."""
    n_test = len(summary["test_acc"])
    n_epochs = len(summary["train_acc"])
    history: dict[str, object] = {
        "seed": seed,
        "generations": n_test,
        "test_acc": [float(value) for value in summary["test_acc"]],
        "test_loss": [float(value) for value in summary.get("test_loss", [])],
        "test_epoch": [],
    }
    if n_test and n_epochs == n_test * epochs_per_generation:
        history["test_epoch"] = [(index + 1) * epochs_per_generation for index in range(n_test)]
    for key in _EPOCH_KEYS:
        history[key] = [float(value) for value in summary[key]]
    for key in _ACTION_KEYS:
        history[key] = list(summary.get(key, []))
    return history


def run_seed(data: Cifar10Data, out_dir: Path, seed: int) -> None:
    """Train one seed with the generation loop. Official test is logged and does not stop the run."""
    if (out_dir / "history.pt").is_file():
        print(f"SKIP seed {seed} history exists at {out_dir / 'history.pt'}", flush=True)
        return
    hp = added_hyperparameters()
    hp["generations"] = GENERATIONS
    hp["epochs"] = EPOCHS
    hp["model_channels"] = CHANNELS
    hp["model_hidden_dim"] = HIDDEN
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(2 if device.type == "cuda" else 4)
    seed_all(seed)
    model = build_model("added", hp)
    train_loader, val_loader, clean_loader = data.loaders(int(hp["batch_size"]))
    train_loader = with_cutout(train_loader)
    test_loader = data.test_loader(int(hp["batch_size"]))
    sim_train, sim_val = sample_loaders(clean_loader, val_loader, int(hp["simulation_set_size"]), seed=seed)
    cfg = running_config_for_group("added", device)
    cfg.generations = GENERATIONS
    cfg.epochs = EPOCHS
    cfg.device = str(device)
    cfg.lr_scheduler = CosineDecayScheduler(GENERATIONS * EPOCHS)
    cfg.simulation_alg = montecarlo_alg
    cfg.simulation_score = SimulationScore(weight_acc=1.0, weight_countW=0.0)
    cfg.stopper = AccuracyStopper(target_accuracy=0.999)
    cfg.criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    print(
        f"EXP008 seed {seed} device {device} generations {GENERATIONS} epochs {EPOCHS} "
        f"channels {CHANNELS} hidden {HIDDEN} smooth {LABEL_SMOOTHING} cutout True",
        flush=True,
    )
    grown, summary = train_generations(
        model, train_loader, val_loader, cfg,
        sim_train_loader=sim_train, sim_val_loader=sim_val, test_loader=test_loader,
    )
    print(f"TEST seed {seed}", [round(float(value), 4) for value in summary["test_acc"]], flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = history_from_generations(summary, seed, EPOCHS)
    torch.save(history, out_dir / "history.pt")
    torch.save(
        {"state_dict": grown.state_dict(), "test_acc": summary["test_acc"], "seed": seed},
        out_dir / "grown.pt",
    )
    print(f"HISTORY -> {out_dir / 'history.pt'} best_test {max(summary['test_acc']):.4f}", flush=True)


def main(argv: list[str] | None = None) -> None:
    """Run the generation-only recipe for the selected seeds."""
    args = parse_args(argv)
    print(f"Exp 008 seeds {', '.join(str(seed) for seed in args.seeds)}.", flush=True)
    print(f"Write target: {RUNS_DIR}", flush=True)
    data = Cifar10Data(DATA_DIR)
    data.prepare()
    for seed in args.seeds:
        run_seed(data, RUNS_DIR / f"seed_{seed}", seed)


if __name__ == "__main__":
    main()
