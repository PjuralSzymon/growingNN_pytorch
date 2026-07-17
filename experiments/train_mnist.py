"""MNIST growingNN run on a minimal convolutional network."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.createsummary import (
    MNIST_HISTORY_FILENAME,
    build_mnist_hyperparameter_folder_name,
)
from experiments.experiments_common import (
    ExperimentDefinition,
    parse_board_cli,
    run_experiment_grid,
)
from growingnn.core.config import DATALOADER_NUM_WORKERS
from growingnn.core.logger import logger

# MNIST grid defaults.
GENERATIONS = [10]
EPOCHS = [30]
BATCH_SIZE = [64]
LR_ALPHA = [0.01]
SIMULATION_TIME = [500.0]
SIMULATION_EPOCHS = [15]
SIMULATION_SET_SIZE = [2000]
TARGET_ACCURACY = [0.99]
SCORE_WEIGHT_ACC = [1.0]
SCORE_WEIGHT_COUNTW = [0.1]
MODEL_CHANNELS = [4]
HIDDEN_LINEAR_SIZE = [16]
GRID_SEEDS = [0]

METAPARAM_KEYS = (
    "dataset",
    "generations",
    "epochs",
    "batch_size",
    "lr_alpha",
    "simulation_time",
    "simulation_epochs",
    "simulation_set_size",
    "target_accuracy",
    "score_weight_acc",
    "score_weight_countw",
    "model_channels",
    "hidden_linear_size",
)
METAPARAM_LISTS = (
    ["mnist"],
    GENERATIONS,
    EPOCHS,
    BATCH_SIZE,
    LR_ALPHA,
    SIMULATION_TIME,
    SIMULATION_EPOCHS,
    SIMULATION_SET_SIZE,
    TARGET_ACCURACY,
    SCORE_WEIGHT_ACC,
    SCORE_WEIGHT_COUNTW,
    MODEL_CHANNELS,
    HIDDEN_LINEAR_SIZE,
)

OUT_DIR = _EXPERIMENT_DIR / "output" / "train_mnist"
DATA_DIR = _EXPERIMENT_DIR / "data" / "mnist"
RUNS_DIR = OUT_DIR / "runs"
NUM_CLASSES = 10
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)


class MNISTData:
    """MNIST loaders; datasets and DataLoaders are built once per process."""

    def __init__(
        self, data_dir: Path, *, num_workers: int = DATALOADER_NUM_WORKERS
    ) -> None:
        self._data_dir = data_dir
        self._num_workers = num_workers
        self._datasets: (
            tuple[datasets.MNIST, datasets.MNIST, datasets.MNIST] | None
        ) = None
        self._loader_cache: dict[int, tuple[DataLoader, DataLoader, DataLoader]] = {}

    @staticmethod
    def _eval_transform() -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(MNIST_MEAN, MNIST_STD),
            ]
        )

    @classmethod
    def _train_transform(cls) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(MNIST_MEAN, MNIST_STD),
            ]
        )

    def prepare(self) -> None:
        if self._datasets is not None:
            return
        self._data_dir.mkdir(parents=True, exist_ok=True)
        download = not (self._data_dir / "MNIST" / "raw").is_dir()
        root = str(self._data_dir)
        self._datasets = (
            datasets.MNIST(
                root,
                train=True,
                download=download,
                transform=self._train_transform(),
            ),
            datasets.MNIST(
                root,
                train=True,
                download=download,
                transform=self._eval_transform(),
            ),
            datasets.MNIST(
                root,
                train=False,
                download=download,
                transform=self._eval_transform(),
            ),
        )
        train, _, val = self._datasets
        logger.info("Loaded MNIST: %s train, %s val", len(train), len(val))

    def loaders(self, batch_size: int) -> tuple[DataLoader, DataLoader, DataLoader]:
        self.prepare()
        if batch_size in self._loader_cache:
            return self._loader_cache[batch_size]
        train, train_clean, val = self._datasets
        kwargs: dict[str, object] = {
            "batch_size": batch_size,
            "num_workers": self._num_workers,
        }
        if self._num_workers > 0:
            kwargs["persistent_workers"] = True
        pin_memory = torch.cuda.is_available()
        loaders = (
            DataLoader(train, shuffle=True, pin_memory=pin_memory, **kwargs),
            DataLoader(val, pin_memory=pin_memory, **kwargs),
            DataLoader(train_clean, shuffle=False, pin_memory=pin_memory, **kwargs),
        )
        self._loader_cache[batch_size] = loaders
        return loaders


class MinimalMnistNet(nn.Module):
    """Stem conv + one hidden conv + two-layer linear head for FX growth actions."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        channels: int = 4,
        hidden_linear_size: int = 16,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.linear = nn.Linear(channels, hidden_linear_size, bias=True)
        self.linear2 = nn.Linear(hidden_linear_size, num_classes, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.adaptive_avg_pool2d(x, 1)
        x = F.relu(self.linear(x.flatten(1)))
        return self.linear2(x)


def _build_model(hp: dict[str, object]) -> nn.Module:
    return MinimalMnistNet(
        channels=int(hp["model_channels"]),
        hidden_linear_size=int(hp["hidden_linear_size"]),
    )


if __name__ == "__main__":
    args = parse_board_cli("train_mnist growingNN experiment")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = MNISTData(DATA_DIR)
    data.prepare()
    definition = ExperimentDefinition(
        name="MNIST",
        runs_dir=RUNS_DIR,
        history_filename=MNIST_HISTORY_FILENAME,
        seeds=GRID_SEEDS,
        folder_name=build_mnist_hyperparameter_folder_name,
        model_factory=_build_model,
        loader_factory=lambda hp: data.loaders(int(hp["batch_size"])),
        board_metadata=lambda hp, folder, seed: (
            f"MNIST minimal | {folder} | seed {seed}",
            "MNIST",
        ),
    )
    grid = (
        dict(zip(METAPARAM_KEYS, combo))
        for combo in itertools.product(*METAPARAM_LISTS)
    )
    executed, skipped = run_experiment_grid(
        definition,
        grid,
        device=device,
        board=args.board,
    )
    print(
        f"Finished {executed} run(s), skipped {skipped} existing run(s) under {RUNS_DIR}"
    )
