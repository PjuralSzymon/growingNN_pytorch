"""MNIST-like growingNN benchmark — extend DATASET_ORDER and GRID lists to add more runs."""

from __future__ import annotations

import importlib
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.error import URLError

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
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

# MedMNIST: pip install medmnist
# EM = EMNIST (balanced); OrganM = OrganAMNIST (axial CT slices).
DATASET_ORDER = (
    "breastm",
    "em",
    "fashionm",
    "mnist",
    "kmnist",
    "organm",
    "pneumoniam",
)

# Locked from 28-run grid summary (Jul 2026): accuracy-first defaults; sweep hidden_linear_size only.
GENERATIONS = [10]              # g10 in all top MNIST/Pneumoniam runs; g20 helped only one BreastM config
EPOCHS = [30]                   # only value tested
BATCH_SIZE = [64]               # only value tested
LR_ALPHA = [0.01]               # all best configs used 0.01; 0.003 not yet validated
SIMULATION_TIME = [500.0]
SIMULATION_EPOCHS = [15]
SIMULATION_SET_SIZE = [2000]
TARGET_ACCURACY = [0.99]
SCORE_WEIGHT_ACC = [1.0]
SCORE_WEIGHT_COUNTW = [0.1]     # +8% mean val_acc vs 0.2 (70.6% vs 62.5%)
MODEL_CHANNELS = [4]            # best mean val_acc across datasets (69.5%); ch3 wins MNIST only
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
GRID_PARAM_KEYS = METAPARAM_KEYS[1:]
GRID_PARAM_LISTS = (
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
DATA_ROOT = _EXPERIMENT_DIR / "data"
RUNS_DIR = OUT_DIR / "runs"
MEDMNIST_SIZE = 28


@dataclass(frozen=True)
class DatasetConfiguration:
    """Metadata and factory functions needed to load one benchmark dataset."""

    key: str
    num_classes: int
    in_channels: int
    build_train: Callable[[Path, bool, bool], Dataset]
    build_eval: Callable[[Path], Dataset]
    is_cached: Callable[[Path], bool]


def _create_image_transform(
    mean: tuple[float, ...],
    std: tuple[float, ...],
    *,
    augment: bool,
) -> transforms.Compose:
    """Create normalization steps with optional training-image augmentation."""
    steps: list[Any] = []
    if augment:
        steps.append(transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)))
    steps.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    return transforms.Compose(steps)


class _ScalarLabelDataset(Dataset):
    """MedMNIST returns shape-(1,) labels; CrossEntropyLoss needs scalar class indices."""

    def __init__(self, base: Dataset) -> None:
        self._base = base

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        image, label = self._base[index]
        if hasattr(label, "__len__") and not isinstance(label, (str, bytes)):
            return image, int(label[0])
        return image, int(label)


class TorchvisionDatasetSource:
    """Create dataset configurations for datasets provided by torchvision."""

    @staticmethod
    def is_downloaded(root: Path, dataset_name: str) -> bool:
        raw = root / dataset_name / "raw"
        return raw.is_dir() and any(raw.iterdir())

    @staticmethod
    def builders(
        dataset_cls: type,
        mean: tuple[float, ...],
        std: tuple[float, ...],
        *,
        emnist_split: str | None = None,
    ) -> tuple[Callable[[Path, bool, bool], Dataset], Callable[[Path], Dataset]]:
        def build_train(root: Path, download: bool, augment: bool) -> Dataset:
            kwargs: dict[str, Any] = {
                "root": str(root),
                "train": True,
                "download": download,
                "transform": _create_image_transform(mean, std, augment=augment),
            }
            if emnist_split is not None:
                kwargs["split"] = emnist_split
            return dataset_cls(**kwargs)

        def build_eval(root: Path) -> Dataset:
            kwargs = {
                "root": str(root),
                "train": False,
                "download": False,
                "transform": _create_image_transform(mean, std, augment=False),
            }
            if emnist_split is not None:
                kwargs["split"] = emnist_split
            return dataset_cls(**kwargs)

        return build_train, build_eval

    @classmethod
    def create_spec(
        cls,
        key: str,
        dataset_cls: type,
        num_classes: int,
        mean: tuple[float, ...],
        std: tuple[float, ...],
        *,
        dataset_name: str,
        emnist_split: str | None = None,
    ) -> DatasetConfiguration:
        build_train, build_eval = cls.builders(
            dataset_cls, mean, std, emnist_split=emnist_split
        )
        return DatasetConfiguration(
            key,
            num_classes,
            1,
            build_train,
            build_eval,
            is_cached=lambda root: cls.is_downloaded(root, dataset_name),
        )


class MedMnistDatasetSource:
    """Create dataset configurations for datasets provided by MedMNIST."""

    @staticmethod
    def is_downloaded(root: Path, dataset_name: str) -> bool:
        return (root / f"{dataset_name}.npz").is_file()

    @staticmethod
    def builders(
        dataset_cls: type,
        mean: tuple[float, ...],
        std: tuple[float, ...],
    ) -> tuple[Callable[[Path, bool, bool], Dataset], Callable[[Path], Dataset]]:
        def build_train(root: Path, download: bool, augment: bool) -> Dataset:
            return _ScalarLabelDataset(
                dataset_cls(
                    split="train",
                    root=str(root),
                    download=download,
                    size=MEDMNIST_SIZE,
                    transform=_create_image_transform(mean, std, augment=augment),
                )
            )

        def build_eval(root: Path) -> Dataset:
            return _ScalarLabelDataset(
                dataset_cls(
                    split="test",
                    root=str(root),
                    download=False,
                    size=MEDMNIST_SIZE,
                    transform=_create_image_transform(mean, std, augment=False),
                )
            )

        return build_train, build_eval

    @classmethod
    def create_spec(
        cls,
        key: str,
        dataset_cls: type,
        num_classes: int,
        in_channels: int,
        mean: tuple[float, ...],
        std: tuple[float, ...],
        *,
        dataset_name: str,
    ) -> DatasetConfiguration:
        build_train, build_eval = cls.builders(dataset_cls, mean, std)
        return DatasetConfiguration(
            key,
            num_classes,
            in_channels,
            build_train,
            build_eval,
            is_cached=lambda root: cls.is_downloaded(root, dataset_name),
        )


def _create_dataset_configurations() -> dict[str, DatasetConfiguration]:
    """Create the runtime loading configuration for every selected benchmark dataset."""
    configurations: dict[str, DatasetConfiguration] = {}
    gray = lambda m, s: ((m,), (s,))
    configurations["mnist"] = TorchvisionDatasetSource.create_spec(
        "mnist", datasets.MNIST, 10, *gray(0.1307, 0.3081), dataset_name="MNIST"
    )
    configurations["fashionm"] = TorchvisionDatasetSource.create_spec(
        "fashionm",
        datasets.FashionMNIST,
        10,
        *gray(0.2860, 0.3530),
        dataset_name="FashionMNIST",
    )
    configurations["kmnist"] = TorchvisionDatasetSource.create_spec(
        "kmnist", datasets.KMNIST, 10, *gray(0.1904, 0.3355), dataset_name="KMNIST"
    )
    configurations["em"] = TorchvisionDatasetSource.create_spec(
        "em",
        datasets.EMNIST,
        47,
        *gray(0.1751, 0.3332),
        dataset_name="EMNIST",
        emnist_split="balanced",
    )

    medmnist = importlib.import_module("medmnist")
    rgb = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    for key, info_key in (
        ("breastm", "breastmnist"),
        ("organm", "organamnist"),
        ("pneumoniam", "pneumoniamnist"),
    ):
        meta = medmnist.INFO[info_key]
        dataset_cls = getattr(medmnist, meta["python_class"])
        channels = int(meta["n_channels"])
        mean, std = rgb if channels == 3 else gray(0.5, 0.5)
        configurations[key] = MedMnistDatasetSource.create_spec(
            key,
            dataset_cls,
            len(meta["label"]),
            channels,
            mean,
            std,
            dataset_name=info_key,
        )
    return configurations


class BenchmarkData:
    """Load one registered dataset once; reuse DataLoaders per batch size."""

    def __init__(
        self,
        spec: DatasetConfiguration,
        root: Path,
        *,
        num_workers: int = DATALOADER_NUM_WORKERS,
    ) -> None:
        self._spec = spec
        self._root = root / spec.key
        self._num_workers = num_workers
        self._datasets: tuple[Dataset, Dataset] | None = None
        self._loader_cache: dict[int, tuple[DataLoader, DataLoader, DataLoader]] = {}

    def prepare(self) -> None:
        if self._datasets is not None:
            return
        self._root.mkdir(parents=True, exist_ok=True)
        download = not self._spec.is_cached(self._root)
        if download:
            logger.info("Downloading %s into %s", self._spec.key, self._root)
        try:
            train = self._spec.build_train(self._root, download, augment=True)
            val = self._spec.build_eval(self._root)
        except URLError as exc:
            raise RuntimeError(
                f"Cannot download dataset '{self._spec.key}' (network/DNS error). "
                f"Connect to the internet and re-run, or place cached files under {self._root}. "
                f"Torchvision sets need {self._root}/<Name>/raw/*.ubyte; "
                f"MedMNIST sets need {self._root}/*.npz."
            ) from exc
        self._datasets = (train, val)
        logger.info("Loaded %s: %s train, %s val", self._spec.key, len(train), len(val))

    def loaders(self, batch_size: int) -> tuple[DataLoader, DataLoader, DataLoader]:
        self.prepare()
        if batch_size in self._loader_cache:
            return self._loader_cache[batch_size]
        train, val = self._datasets
        kwargs: dict[str, object] = {"batch_size": batch_size, "num_workers": self._num_workers}
        pin = torch.cuda.is_available()
        clean_train = self._spec.build_train(self._root, download=False, augment=False)
        loaders = (
            DataLoader(train, shuffle=True, pin_memory=pin, **kwargs),
            DataLoader(val, pin_memory=pin, **kwargs),
            DataLoader(clean_train, shuffle=False, pin_memory=pin, **kwargs),
        )
        self._loader_cache[batch_size] = loaders
        return loaders


class SmallMnistNet(nn.Module):
    """Stem conv + one hidden conv + two-layer linear head for FX growth actions."""

    def __init__(self, num_classes: int, channels: int, in_channels: int = 1, hidden_linear_size: int = 16) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, 3, padding=1, bias=False)
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


def _iter_grid_hyperparameters() -> list[dict[str, object]]:
    grid: list[dict[str, object]] = []
    for combo in itertools.product(*GRID_PARAM_LISTS):
        base = dict(zip(GRID_PARAM_KEYS, combo))
        for dataset in DATASET_ORDER:
            grid.append({"dataset": dataset, **base})
    return grid


if __name__ == "__main__":
    args = parse_board_cli("train_mnist growingNN experiment")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_configurations = _create_dataset_configurations()
    definition = ExperimentDefinition(
        name="MNIST benchmarks",
        runs_dir=RUNS_DIR,
        history_filename=MNIST_HISTORY_FILENAME,
        seeds=GRID_SEEDS,
        folder_name=build_mnist_hyperparameter_folder_name,
        model_factory=lambda hp: SmallMnistNet(
            dataset_configurations[str(hp["dataset"])].num_classes,
            int(hp["model_channels"]),
            in_channels=dataset_configurations[str(hp["dataset"])].in_channels,
            hidden_linear_size=int(hp["hidden_linear_size"]),
        ),
        loader_factory=lambda hp: BenchmarkData(
            dataset_configurations[str(hp["dataset"])], DATA_ROOT
        ).loaders(int(hp["batch_size"])),
        board_metadata=lambda hp, folder, seed: (
            f"{str(hp['dataset']).upper()} | {folder} | seed {seed}",
            str(hp["dataset"]).upper(),
        ),
    )
    executed, skipped = run_experiment_grid(
        definition,
        _iter_grid_hyperparameters(),
        device=device,
        board=args.board,
    )
    print(f"Finished {executed} run(s), skipped {skipped} existing run(s) under {RUNS_DIR}")
