"""CIFAR-10 growingNN run on a minimal ResNet-style backbone."""

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

from experiments.createsummary import HISTORY_FILENAME, build_hyperparameter_folder_name
from experiments.experiments_common import (
    ExperimentDefinition,
    parse_board_cli,
    require_cuda,
    run_experiment_grid,
)
from growingnn.core.config import DATALOADER_NUM_WORKERS
from growingnn.core.logger import logger

# ~24 configs x 3 seeds = 72 runs, ~30-44 h on 8 GB GPU
GENERATIONS = [10, 20]
EPOCHS = [30]
BATCH_SIZE = [64]
LR_ALPHA = [0.01]
SIMULATION_TIME = [500.0]
SIMULATION_EPOCHS = [15]
SIMULATION_SET_SIZE = [2000]
TARGET_ACCURACY = [0.99]
SCORE_WEIGHT_ACC = [1.0]
SCORE_WEIGHT_COUNTW = [0.2]
MODEL_CHANNELS = [32]
MODEL_HIDDEN_DIM = [256]
MODEL_NUM_BLOCKS = [1]
GRID_REPEAT_SEEDS = [110]

METAPARAM_KEYS = (
    "generations", "epochs", "batch_size", "lr_alpha", "simulation_time",
    "simulation_epochs", "simulation_set_size", "target_accuracy",
    "score_weight_acc", "score_weight_countw", "model_channels",
    "model_hidden_dim", "model_num_blocks",
)
METAPARAM_LISTS = (
    GENERATIONS, EPOCHS, BATCH_SIZE, LR_ALPHA, SIMULATION_TIME, SIMULATION_EPOCHS,
    SIMULATION_SET_SIZE, TARGET_ACCURACY, SCORE_WEIGHT_ACC, SCORE_WEIGHT_COUNTW,
    MODEL_CHANNELS, MODEL_HIDDEN_DIM, MODEL_NUM_BLOCKS,
)

OUT_DIR = _EXPERIMENT_DIR / "output" / "train_cifar10"
DATA_DIR = _EXPERIMENT_DIR / "data" / "cifar10"
RUNS_DIR = OUT_DIR / "runs"
NUM_CLASSES = 10
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


class Cifar10Data:
    """CIFAR-10 loaders; datasets and DataLoaders are built once per process."""

    def __init__(self, data_dir: Path, *, num_workers: int = DATALOADER_NUM_WORKERS) -> None:
        self._data_dir = data_dir
        self._num_workers = num_workers
        self._datasets: tuple[datasets.CIFAR10, datasets.CIFAR10, datasets.CIFAR10] | None = None
        self._loader_cache: dict[int, tuple[DataLoader, DataLoader, DataLoader]] = {}

    @staticmethod
    def _eval_transform() -> transforms.Compose:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])

    @classmethod
    def _train_transform(cls) -> transforms.Compose:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])

    def prepare(self) -> None:
        if self._datasets is not None:
            return
        self._data_dir.mkdir(parents=True, exist_ok=True)
        download = not (self._data_dir / "cifar-10-batches-py").is_dir()
        root = str(self._data_dir)
        self._datasets = (
            datasets.CIFAR10(root, train=True, download=download, transform=self._train_transform()),
            datasets.CIFAR10(root, train=True, download=download, transform=self._eval_transform()),
            datasets.CIFAR10(root, train=False, download=download, transform=self._eval_transform()),
        )
        train, _, val = self._datasets
        logger.info("Loaded CIFAR-10: %s train, %s val", len(train), len(val))

    def loaders(self, batch_size: int) -> tuple[DataLoader, DataLoader, DataLoader]:
        self.prepare()
        if batch_size in self._loader_cache:
            return self._loader_cache[batch_size]
        train, train_clean, val = self._datasets
        kwargs: dict[str, object] = {"batch_size": batch_size, "num_workers": self._num_workers}
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


class MinimalBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class MinimalCifarNet(nn.Module):
    """Tiny ResNet for CIFAR-10: stem + 1 or 2 residual blocks."""

    @staticmethod
    def _block_specs(channels: int, hidden_dim: int, num_blocks: int) -> list[tuple[int, int]]:
        if num_blocks == 1:
            return [(hidden_dim, 2)]
        if num_blocks == 2:
            return [(channels, 1), (hidden_dim, 2)]
        raise ValueError(f"model_num_blocks must be 1 or 2, got {num_blocks}")

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        channels: int = 8,
        hidden_dim: int = 32,
        num_blocks: int = 1,
    ) -> None:
        super().__init__()
        if num_blocks not in (1, 2):
            raise ValueError(f"model_num_blocks must be 1 or 2, got {num_blocks}")
        self.num_blocks = num_blocks
        self.conv1 = nn.Conv2d(3, channels, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        in_planes = channels
        pool_size = 32
        for i, (out_planes, stride) in enumerate(
            self._block_specs(channels, hidden_dim, num_blocks), start=1
        ):
            setattr(self, f"layer{i}", MinimalBasicBlock(in_planes, out_planes, stride))
            in_planes = out_planes
            pool_size //= stride
        self._pool_size = pool_size
        self.linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        for i in range(1, self.num_blocks + 1):
            x = getattr(self, f"layer{i}")(x)
        x = F.avg_pool2d(x, self._pool_size)
        return self.linear(torch.flatten(x, 1))


def _build_model(hp: dict[str, object]) -> nn.Module:
    return MinimalCifarNet(
        channels=int(hp["model_channels"]),
        hidden_dim=int(hp["model_hidden_dim"]),
        num_blocks=int(hp["model_num_blocks"]),
    )


if __name__ == "__main__":
    args = parse_board_cli("train_cifar10 minimal growingNN experiment")
    device = torch.device("cuda")
    require_cuda(device)
    data = Cifar10Data(DATA_DIR)
    data.prepare()
    definition = ExperimentDefinition(
        name="CIFAR-10",
        runs_dir=RUNS_DIR,
        history_filename=HISTORY_FILENAME,
        seeds=GRID_REPEAT_SEEDS,
        folder_name=build_hyperparameter_folder_name,
        model_factory=_build_model,
        loader_factory=lambda hp: data.loaders(int(hp["batch_size"])),
        board_metadata=lambda _hp, folder, seed: (
            f"CIFAR-10 minimal | {folder} | seed {seed}",
            "CIFAR-10",
        ),
        save_fx_graphs=True,
    )
    grid = (
        dict(zip(METAPARAM_KEYS, combo))
        for combo in itertools.product(*METAPARAM_LISTS)
    )
    executed, skipped = run_experiment_grid(
        definition, grid, device=device, board=args.board
    )
    print(f"Finished {executed} run(s), skipped {skipped} existing run(s) under {RUNS_DIR}")
