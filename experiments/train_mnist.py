"""MNIST-like growingNN benchmark — extend DATASET_ORDER and GRID lists to add more runs."""

from __future__ import annotations

import argparse
import importlib
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.error import URLError

import matplotlib.pyplot as plt
import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from growingnn.board import ExperimentBoard
from growingnn.core.config import DATALOADER_NUM_WORKERS, RunningConfig
from growingnn.core.logger import logger
import growingnn.simulation.simulation_algorithms.montecarlo_alg as montecarlo_alg
from growingnn.simulation.score_functions.simulation_score import SimulationScore
from growingnn.simulation.simulation_scheduler import SchedulerMode, SimulationScheduler
from growingnn.simulation.simulation_set import sample_loaders
from growingnn.training.lr_scheduler import LearningRateScheduler, ScheduleMode
from growingnn.training.stoppers import AccuracyStopper
from growingnn.training.trainer import train_generations
from growingnn.utils.fx import GraphStructureQuery

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

# model_channels sets initial size (~9c²+19c params): 2~74, 3~138, 5~320, 10~1090
GENERATIONS = [10, 20]
EPOCHS = [30]
BATCH_SIZE = [64]
LR_ALPHA = [0.01]
SIMULATION_TIME = [500.0]
SIMULATION_EPOCHS = [15]
SIMULATION_SET_SIZE = [2000]
TARGET_ACCURACY = [0.99]
SCORE_WEIGHT_ACC = [1.0]
SCORE_WEIGHT_COUNTW = [0.1, 0.2]
MODEL_CHANNELS = [3,4]
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
)
# Grid sweeps hyperparameters outermost, datasets innermost (all datasets per config).
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
)

OUT_DIR = _EXPERIMENT_DIR / "output" / "train_mnist"
DATA_ROOT = _EXPERIMENT_DIR / "data"
RUNS_DIR = OUT_DIR / "runs"
HISTORY_FILE = "train_mnist_history.pt"
METRIC_KEYS = ("train_loss", "train_acc", "val_loss", "val_acc", "lr", "param_count")
SPATIAL = 28


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    num_classes: int
    in_channels: int
    mean: tuple[float, ...]
    std: tuple[float, ...]
    build_train: Callable[[Path, bool, bool], Dataset]
    build_eval: Callable[[Path], Dataset]
    is_cached: Callable[[Path], bool]


def _tv_cache_ready(root: Path, torchvision_name: str) -> bool:
    raw = root / torchvision_name / "raw"
    return raw.is_dir() and any(raw.iterdir())


def _medmnist_cache_ready(root: Path, npz_name: str) -> bool:
    return (root / npz_name).is_file()


def _gray_norm(mean: float, std: float) -> tuple[tuple[float, ...], tuple[float, ...]]:
    return (mean,), (std,)


def _rgb_norm() -> tuple[tuple[float, ...], tuple[float, ...]]:
    return (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)


def _torchvision_transform(
    mean: tuple[float, ...], std: tuple[float, ...], *, train: bool
) -> transforms.Compose:
    steps: list[Any] = []
    if train:
        steps.append(transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)))
    steps.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    return transforms.Compose(steps)


def _tv_builder(
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
            "transform": _torchvision_transform(mean, std, train=augment),
        }
        if emnist_split is not None:
            kwargs["split"] = emnist_split
        return dataset_cls(**kwargs)

    def build_eval(root: Path) -> Dataset:
        kwargs = {
            "root": str(root),
            "train": False,
            "download": False,
            "transform": _torchvision_transform(mean, std, train=False),
        }
        if emnist_split is not None:
            kwargs["split"] = emnist_split
        return dataset_cls(**kwargs)

    return build_train, build_eval


def _medmnist_builder(
    dataset_cls: type,
    mean: tuple[float, ...],
    std: tuple[float, ...],
) -> tuple[Callable[[Path, bool, bool], Dataset], Callable[[Path], Dataset]]:
    def _transform(augment: bool) -> transforms.Compose:
        return _torchvision_transform(mean, std, train=augment)

    def _wrap(dataset: Dataset) -> Dataset:
        return _ScalarLabelDataset(dataset)

    def build_train(root: Path, download: bool, augment: bool) -> Dataset:
        return _wrap(
            dataset_cls(
                split="train",
                root=str(root),
                download=download,
                size=SPATIAL,
                transform=_transform(augment),
            )
        )

    def build_eval(root: Path) -> Dataset:
        return _wrap(
            dataset_cls(
                split="test",
                root=str(root),
                download=False,
                size=SPATIAL,
                transform=_transform(False),
            )
        )

    return build_train, build_eval


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


def _register_torchvision(
    specs: dict[str, DatasetSpec],
    key: str,
    dataset_cls: type,
    num_classes: int,
    mean: tuple[float, ...],
    std: tuple[float, ...],
    *,
    torchvision_name: str,
    emnist_split: str | None = None,
) -> None:
    build_train, build_eval = _tv_builder(dataset_cls, mean, std, emnist_split=emnist_split)
    specs[key] = DatasetSpec(
        key,
        num_classes,
        1,
        mean,
        std,
        build_train,
        build_eval,
        is_cached=lambda root, name=torchvision_name: _tv_cache_ready(root, name),
    )


def _build_dataset_registry() -> dict[str, DatasetSpec]:
    specs: dict[str, DatasetSpec] = {}
    _register_torchvision(
        specs, "mnist", datasets.MNIST, 10, *_gray_norm(0.1307, 0.3081), torchvision_name="MNIST"
    )
    _register_torchvision(
        specs,
        "fashionm",
        datasets.FashionMNIST,
        10,
        *_gray_norm(0.2860, 0.3530),
        torchvision_name="FashionMNIST",
    )
    _register_torchvision(
        specs, "kmnist", datasets.KMNIST, 10, *_gray_norm(0.1904, 0.3355), torchvision_name="KMNIST"
    )
    _register_torchvision(
        specs,
        "em",
        datasets.EMNIST,
        47,
        *_gray_norm(0.1751, 0.3332),
        torchvision_name="EMNIST",
        emnist_split="balanced",
    )

    medmnist = importlib.import_module("medmnist")
    info = medmnist.INFO
    for key, info_key in (
        ("breastm", "breastmnist"),
        ("organm", "organamnist"),
        ("pneumoniam", "pneumoniamnist"),
        ("retinam", "retinamnist"),
    ):
        meta = info[info_key]
        dataset_cls = getattr(medmnist, meta["python_class"])
        channels = int(meta["n_channels"])
        num_classes = len(meta["label"])
        mean, std = _rgb_norm() if channels == 3 else _gray_norm(0.5, 0.5)
        build_train, build_eval = _medmnist_builder(dataset_cls, mean, std)
        npz_name = f"{info_key}.npz"
        specs[key] = DatasetSpec(
            key,
            num_classes,
            channels,
            mean,
            std,
            build_train,
            build_eval,
            is_cached=lambda root, npz=npz_name: _medmnist_cache_ready(root, npz),
        )
    return specs


DATASETS = _build_dataset_registry()


class BenchmarkData:
    """Load one registered dataset once; reuse DataLoaders per batch size."""

    def __init__(self, spec: DatasetSpec, root: Path, *, num_workers: int = DATALOADER_NUM_WORKERS) -> None:
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


def estimate_small_mnist_params(channels: int, num_classes: int = 10) -> int:
    """Return parameter count for SmallMnistNet (stem + hidden conv + adaptive-pool head)."""
    return 9 * channels + 9 * channels * channels + num_classes * channels


class SmallMnistNet(nn.Module):
    """Stem conv + one hidden conv so FX actions can attach (~138 params at channels=3)."""

    def __init__(self, num_classes: int, channels: int, in_channels: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.linear = nn.Linear(channels, num_classes, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.adaptive_avg_pool2d(x, 1)
        return self.linear(x.flatten(1))


def _folder_name(hp: dict[str, object]) -> str:
    return (
        f"ds{hp['dataset']}_g{hp['generations']}_ep{hp['epochs']}_bs{hp['batch_size']}"
        f"_lr{hp['lr_alpha']}_wcw{hp['score_weight_countw']}_ch{hp['model_channels']}"
    )


def _run_dir(hp: dict[str, object], seed: int) -> Path:
    return RUNS_DIR / _folder_name(hp) / f"seed_{seed}"


def _plot_metric(values: list[float], name: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(values) + 1), values)
    ax.set_xlabel("step")
    ax.set_ylabel(name)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _run_once(hp: dict[str, object], *, seed: int, device: torch.device, board: bool) -> None:
    spec = DATASETS[str(hp["dataset"])]
    run_dir = _run_dir(hp, seed)
    if run_dir.exists():
        logger.info("Skip existing %s seed %s", _folder_name(hp), seed)
        return
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    data = BenchmarkData(spec, DATA_ROOT)
    model = SmallMnistNet(
        spec.num_classes,
        int(hp["model_channels"]),
        in_channels=spec.in_channels,
    )
    gm = fx.symbolic_trace(model)
    params_before = GraphStructureQuery.get_amount_of_parameters(gm)
    logger.info(
        "Run %s seed %s params=%s -> %s",
        _folder_name(hp),
        seed,
        params_before,
        run_dir,
    )

    board_writer = (
        ExperimentBoard(
            run_dir / "board",
            experiment_name=f"{spec.key.upper()} | {_folder_name(hp)} | seed {seed}",
            dataset=spec.key.upper(),
            device=str(device),
        )
        if board
        else None
    )
    cfg = RunningConfig(
        generations=int(hp["generations"]),
        epochs=int(hp["epochs"]),
        device=device,
        lr_scheduler=LearningRateScheduler(ScheduleMode.PROGRESSIVE_PARABOLIC, alpha=float(hp["lr_alpha"])),
        simulation_alg=montecarlo_alg,
        simulation_scheduler=SimulationScheduler(
            SchedulerMode.ALWAYS,
            simulation_time=float(hp["simulation_time"]),
            simulation_epochs=int(hp["simulation_epochs"]),
        ),
        stopper=AccuracyStopper(target_accuracy=float(hp["target_accuracy"])),
        simulation_score=SimulationScore(
            weight_acc=float(hp["score_weight_acc"]),
            weight_countW=float(hp["score_weight_countw"]),
        ),
        simulation_set_size=int(hp["simulation_set_size"]),
        criterion=nn.CrossEntropyLoss(),
        quiet=False,
        enable_experiment_board=board,
        experiment_board=board_writer,
    )

    train_loader, val_loader, clean_train = data.loaders(int(hp["batch_size"]))
    sim_train, sim_val = sample_loaders(
        clean_train, val_loader, int(hp["simulation_set_size"]), seed=seed
    )
    gm, summary = train_generations(
        gm, train_loader, val_loader, cfg, sim_train_loader=sim_train, sim_val_loader=sim_val
    )

    history = {k: summary[k] for k in METRIC_KEYS}
    torch.save(history, run_dir / HISTORY_FILE)
    for key in METRIC_KEYS:
        _plot_metric(history[key], key, run_dir / f"{key}.png")
    params_after = GraphStructureQuery.get_amount_of_parameters(gm)
    logger.info(
        "Done %s seed %s val_acc=%.4f params %s -> %s",
        _folder_name(hp),
        seed,
        max(summary["val_acc"]),
        params_before,
        params_after,
    )


def _parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="train_mnist growingNN experiment")
    parser.add_argument(
        "--board",
        choices=("true", "false"),
        default="true",
        help="Write GrowingNN Board artifacts under each run's board/ folder (default: true)",
    )
    ns = parser.parse_args()
    ns.board = ns.board == "true"
    return ns


def _iter_grid_hyperparameters() -> list[dict[str, object]]:
    """One hyperparameter config, then every dataset; repeat for next config."""
    grid: list[dict[str, object]] = []
    for combo in itertools.product(*GRID_PARAM_LISTS):
        base = dict(zip(GRID_PARAM_KEYS, combo))
        for dataset in DATASET_ORDER:
            grid.append({"dataset": dataset, **base})
    return grid


if __name__ == "__main__":
    args = _parse_cli()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    done = 0
    for hp in _iter_grid_hyperparameters():
        for seed in GRID_SEEDS:
            _run_once(hp, seed=seed, device=device, board=args.board)
            done += 1
    print(f"Finished {done} scheduled run(s) under {RUNS_DIR}")
