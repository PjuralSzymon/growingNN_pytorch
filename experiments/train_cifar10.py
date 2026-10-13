"""CIFAR-10 growingNN run on a tiny plain convolutional starter."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
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
from growingnn.simulation.simulation_schedulers import (
    AlwaysSimulationScheduler,
    NeverSimulationScheduler,
    SlopeEstimationSimulationScheduler,
)
from growingnn.training.lr_scheduler_action import (
    ActionLearningRateScheduler,
    LearningRateScheduler,
    ScheduleMode,
    mark_warmup_schedule_as_fully_complete,
)
from growingnn.training.lr_scheduler_global import build_composed_learning_rate_scheduler

# One configuration per loop iteration. Folder names record momentum, decay, and scheduler.
GENERATIONS = [8]
EPOCHS = [10]
BATCH_SIZE = [128]
LR_ALPHA = [0.01]
SIMULATION_TIME = [90.0]
SIMULATION_EPOCHS = [8]
SIMULATION_SET_SIZE = [2000]
TARGET_ACCURACY = [0.99]
SCORE_WEIGHT_ACC = [1.0]
SCORE_WEIGHT_COUNTW = [0.0]
MODEL_CHANNELS = [32]
MODEL_HIDDEN_DIM = [64]
MOMENTUM = [0.9]
WEIGHT_DECAY = [0.0005]
SCHEDULER = ["slope3"]
LR_MODE = ["cosine"]
PLAIN_GROWTH = [True]
GRID_REPEAT_SEEDS = [110]

METAPARAM_KEYS = (
    "generations", "epochs", "batch_size", "lr_alpha", "simulation_time",
    "simulation_epochs", "simulation_set_size", "target_accuracy",
    "score_weight_acc", "score_weight_countw", "model_channels",
    "model_hidden_dim", "momentum", "weight_decay", "scheduler", "lr_mode",
    "plain_growth",
)
METAPARAM_LISTS = (
    GENERATIONS, EPOCHS, BATCH_SIZE, LR_ALPHA, SIMULATION_TIME, SIMULATION_EPOCHS,
    SIMULATION_SET_SIZE, TARGET_ACCURACY, SCORE_WEIGHT_ACC, SCORE_WEIGHT_COUNTW,
    MODEL_CHANNELS, MODEL_HIDDEN_DIM, MOMENTUM, WEIGHT_DECAY, SCHEDULER, LR_MODE,
    PLAIN_GROWTH,
)

OUT_DIR = _EXPERIMENT_DIR / "output" / "train_cifar10"
DATA_DIR = _EXPERIMENT_DIR / "data" / "cifar10"
RUNS_DIR = OUT_DIR / "runs"
NUM_CLASSES = 10
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
TRAIN_SIZE = 45_000
VAL_SIZE = 5_000


class Cifar10Data:
    """CIFAR-10 loaders; datasets and DataLoaders are built once per process."""

    def __init__(self, data_dir: Path, *, num_workers: int = DATALOADER_NUM_WORKERS) -> None:
        self._data_dir = data_dir
        self._num_workers = num_workers
        self._datasets: tuple[Subset, Subset, Subset, datasets.CIFAR10] | None = None
        self._loader_cache: dict[int, tuple[DataLoader, DataLoader, DataLoader]] = {}
        self._test_loader_cache: dict[int, DataLoader] = {}

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
        download = not (self._data_dir / "cifar-10-batches-py" / "data_batch_1").is_file()
        root = str(self._data_dir)
        augmented = datasets.CIFAR10(
            root, train=True, download=download, transform=self._train_transform(),
        )
        clean = datasets.CIFAR10(
            root, train=True, download=False, transform=self._eval_transform(),
        )
        test = datasets.CIFAR10(
            root, train=False, download=False, transform=self._eval_transform(),
        )
        clean_train, val = random_split(
            clean,
            [TRAIN_SIZE, VAL_SIZE],
            generator=torch.Generator().manual_seed(0),
        )
        train = Subset(augmented, clean_train.indices)
        self._datasets = (train, val, clean_train, test)
        logger.info(
            "Loaded CIFAR-10: %s train, %s val, %s test",
            len(train), len(val), len(test),
        )

    def loaders(self, batch_size: int) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Return augmented train, held-out val, and clean train. The official test set is excluded."""
        self.prepare()
        if batch_size in self._loader_cache:
            return self._loader_cache[batch_size]
        train, val, train_clean, _test = self._datasets
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

    def test_loader(self, batch_size: int) -> DataLoader:
        """Load the official test split only for logging, not for search or the stopper."""
        self.prepare()
        if batch_size in self._test_loader_cache:
            return self._test_loader_cache[batch_size]
        test = self._datasets[3]
        pin_memory = torch.cuda.is_available()
        loader = DataLoader(
            test,
            batch_size=batch_size,
            num_workers=self._num_workers,
            pin_memory=pin_memory,
        )
        self._test_loader_cache[batch_size] = loader
        return loader


class MinimalCifarNet(nn.Module):
    """Stem conv + one hidden conv + two-layer linear head for FX growth actions."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        channels: int = 8,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.linear = nn.Linear(channels, hidden_dim, bias=True)
        self.linear2 = nn.Linear(hidden_dim, num_classes, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.adaptive_avg_pool2d(x, 1)
        x = F.relu(self.linear(x.flatten(1)))
        return self.linear2(x)


def _build_model(hp: dict[str, object]) -> nn.Module:
    return MinimalCifarNet(
        channels=int(hp["model_channels"]),
        hidden_dim=int(hp["model_hidden_dim"]),
    )


def learning_rate_scheduler_factory(hp: dict[str, object]) -> LearningRateScheduler:
    """Build the CIFAR LR schedule. Logistic warmup is primed so generation 0 starts at alpha."""
    alpha = float(hp["lr_alpha"])
    epochs = int(hp["epochs"])
    if hp.get("lr_mode") == "cosine":
        return build_composed_learning_rate_scheduler(
            "cosine",
            total_epochs=int(hp["generations"]) * epochs,
            initial_lr=alpha,
            warmup_iterations=epochs,
            k=10.0,
        )
    if hp.get("lr_mode") == "logistic":
        scheduler = ActionLearningRateScheduler(
            ScheduleMode.WARMUP_LOGISTIC,
            alpha=alpha,
            warmup_iterations=epochs,
            k=10.0,
        )
        mark_warmup_schedule_as_fully_complete(scheduler._schedule)
        return scheduler
    return ActionLearningRateScheduler(ScheduleMode.PROGRESSIVE_PARABOLIC, alpha=alpha)


def simulation_scheduler_factory(hp: dict[str, object]):
    """Build never, always, or 3-degree slope search for one CIFAR run."""
    simulation_time = float(hp["simulation_time"])
    simulation_epochs = int(hp["simulation_epochs"])
    name = str(hp.get("scheduler", "always"))
    if name == "never":
        return NeverSimulationScheduler(simulation_time, simulation_epochs)
    if name == "slope3":
        return SlopeEstimationSimulationScheduler(
            simulation_time,
            simulation_epochs,
            angle_threshold=3.0,
        )
    return AlwaysSimulationScheduler(simulation_time, simulation_epochs)


def _with_scheduler_factories(hp: dict[str, object]) -> dict[str, object]:
    full = dict(hp)
    full["lr_scheduler_factory"] = learning_rate_scheduler_factory
    full["simulation_scheduler_factory"] = simulation_scheduler_factory
    return full


if __name__ == "__main__":
    args = parse_board_cli("train_cifar10 plain growingNN experiment")
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
        test_loader_factory=lambda hp: data.test_loader(int(hp["batch_size"])),
        board_metadata=lambda _hp, folder, seed: (
            f"CIFAR-10 plain | {folder} | seed {seed}",
            "CIFAR-10",
        ),
        save_fx_graphs=False,
    )
    grid = (
        _with_scheduler_factories(dict(zip(METAPARAM_KEYS, combo)))
        for combo in itertools.product(*METAPARAM_LISTS)
    )
    executed, skipped = run_experiment_grid(
        definition, grid, device=device, board=args.board
    )
    print(f"Finished {executed} run(s), skipped {skipped} existing run(s) under {RUNS_DIR}")
