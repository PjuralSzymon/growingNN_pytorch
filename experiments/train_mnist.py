"""MNIST growingNN experiment — extend DATASETS and GRID lists to add more runs."""

from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
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

# model_channels sets initial size (~9c²+19c params): 2~74, 3~138, 5~320, 10~1090
GENERATIONS = [10, 20]
EPOCHS = [30]
BATCH_SIZE = [64]
LR_ALPHA = [0.01]
SIMULATION_TIME = [50.0]
SIMULATION_EPOCHS = [15]
SIMULATION_SET_SIZE = [2000]
TARGET_ACCURACY = [0.99]
SCORE_WEIGHT_ACC = [1.0]
SCORE_WEIGHT_COUNTW = [0.1, 0.2]
MODEL_CHANNELS = [3]
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
)

OUT_DIR = _EXPERIMENT_DIR / "output" / "train_mnist"
DATA_ROOT = _EXPERIMENT_DIR / "data"
RUNS_DIR = OUT_DIR / "runs"
HISTORY_FILE = "train_mnist_history.pt"
METRIC_KEYS = ("train_loss", "train_acc", "val_loss", "val_acc", "lr", "param_count")
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081
MNIST_INPUT_SHAPE = (1, 28, 28)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    num_classes: int
    input_shape: tuple[int, int, int]
    mean: float
    std: float

    @property
    def in_channels(self) -> int:
        return self.input_shape[0]

    @property
    def spatial(self) -> int:
        return self.input_shape[1]


DATASETS: dict[str, DatasetSpec] = {
    "mnist": DatasetSpec("mnist", 10, MNIST_INPUT_SHAPE, MNIST_MEAN, MNIST_STD),
}


class MnistData:
    """Load one registered dataset once; reuse DataLoaders per batch size."""

    def __init__(self, spec: DatasetSpec, root: Path, *, num_workers: int = DATALOADER_NUM_WORKERS) -> None:
        self._spec = spec
        self._root = root / spec.name
        self._num_workers = num_workers
        self._datasets: tuple[datasets.MNIST, datasets.MNIST] | None = None
        self._loader_cache: dict[int, tuple[DataLoader, DataLoader, DataLoader]] = {}

    def _transform(self, train: bool) -> transforms.Compose:
        steps = [transforms.ToTensor()]
        if train:
            steps.insert(0, transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)))
        steps.append(transforms.Normalize((self._spec.mean,), (self._spec.std,)))
        return transforms.Compose(steps)

    def prepare(self) -> None:
        if self._datasets is not None:
            return
        self._root.mkdir(parents=True, exist_ok=True)
        download = not (self._root / "MNIST" / "raw").is_dir()
        root = str(self._root)
        train = datasets.MNIST(root, train=True, download=download, transform=self._transform(True))
        val = datasets.MNIST(root, train=False, download=download, transform=self._transform(False))
        self._datasets = (train, val)
        logger.info("Loaded %s: %s train, %s val", self._spec.name, len(train), len(val))

    def loaders(self, batch_size: int) -> tuple[DataLoader, DataLoader, DataLoader]:
        self.prepare()
        if batch_size in self._loader_cache:
            return self._loader_cache[batch_size]
        train, val = self._datasets
        kwargs: dict[str, object] = {"batch_size": batch_size, "num_workers": self._num_workers}
        pin = torch.cuda.is_available()
        clean_train = datasets.MNIST(
            str(self._root),
            train=True,
            download=False,
            transform=self._transform(False),
        )
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

    def __init__(self, num_classes: int = 10, channels: int = 3) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, channels, 3, padding=1, bias=False)
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


def _install_shape_probe(spec: DatasetSpec) -> None:
    from growingnn.utils.fx.graph_analysis import LayerShapeAnalyser

    spatial = spec.spatial

    @staticmethod
    def _probe(gm: fx.GraphModule) -> torch.Tensor | None:
        if not any(n.op == "placeholder" for n in gm.graph.nodes):
            return None
        try:
            p0 = next(gm.parameters())
            device, dtype = p0.device, p0.dtype
        except StopIteration:
            device, dtype = torch.device("cpu"), torch.float32
        for mod in gm.modules():
            if isinstance(mod, nn.Linear):
                return torch.randn(1, mod.in_features, device=device, dtype=dtype)
            if isinstance(mod, nn.modules.conv._ConvNd):
                return torch.randn(1, mod.in_channels, spatial, spatial, device=device, dtype=dtype)
        return torch.randn(1, *spec.input_shape, device=device, dtype=dtype)

    LayerShapeAnalyser.default_example_input = _probe


def _run_once(hp: dict[str, object], *, seed: int, device: torch.device, board: bool) -> None:
    spec = DATASETS[str(hp["dataset"])]
    _install_shape_probe(spec)
    run_dir = _run_dir(hp, seed)
    if run_dir.exists():
        logger.info("Skip existing %s seed %s", _folder_name(hp), seed)
        return
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    data = MnistData(spec, DATA_ROOT)
    model = SmallMnistNet(spec.num_classes, int(hp["model_channels"]))
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
            experiment_name=f"MNIST | {_folder_name(hp)} | seed {seed}",
            dataset=spec.name.upper(),
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


if __name__ == "__main__":
    args = _parse_cli()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    grid = [dict(zip(METAPARAM_KEYS, combo)) for combo in itertools.product(*METAPARAM_LISTS)]
    done = 0
    for hp in grid:
        for seed in GRID_SEEDS:
            _run_once(hp, seed=seed, device=device, board=args.board)
            done += 1
    print(f"Finished {done} scheduled run(s) under {RUNS_DIR}")
