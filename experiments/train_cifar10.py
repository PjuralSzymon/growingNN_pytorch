"""CIFAR-10 growingNN run on a minimal ResNet-style backbone."""

from __future__ import annotations

import argparse
import itertools
import sys
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
if str(_EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENT_DIR))

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
from growingnn.utils.fx_graph_drawer import draw_filtered_fx_graph, draw_torch_fx_graph
from createsummary import HISTORY_FILENAME, build_hyperparameter_folder_name, run_dir_for_seed

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
CIFAR_INPUT_SHAPE = (3, 32, 32)
METRIC_KEYS = ("train_loss", "train_acc", "val_loss", "val_acc", "lr", "param_count")


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


def _draw_graphs(run_dir: Path, generation: int, gm: fx.GraphModule) -> None:
    draw_filtered_fx_graph(gm, str(run_dir / f"fx_graph_simplified{generation}"), fmt="pdf")
    draw_torch_fx_graph(gm, str(run_dir / f"fx_graph{generation}"), fmt="pdf")


def _plot_metric(values: list[float], name: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(values) + 1), values)
    ax.set_xlabel("step")
    ax.set_ylabel(name)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="train_cifar10 minimal growingNN experiment")
    parser.add_argument(
        "--board",
        choices=("true", "false"),
        default="true",
        help="Write GrowingNN Board artifacts under each run's board/ folder (default: true)",
    )
    args = parser.parse_args()
    args.board = args.board == "true"
    return args


def _run_once(
    hp: dict[str, object],
    *,
    seed: int,
    device: torch.device,
    board: bool,
    data: Cifar10Data,
) -> None:
    folder = build_hyperparameter_folder_name(hp)
    run_dir = run_dir_for_seed(RUNS_DIR, folder, seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Run %s seed %s -> %s", folder, seed, run_dir)

    gm = fx.symbolic_trace(_build_model(hp))
    params_before = GraphStructureQuery.get_amount_of_parameters(gm)
    _draw_graphs(run_dir, 0, gm)

    board_writer = (
        ExperimentBoard(
            run_dir / "board",
            experiment_name=f"CIFAR-10 minimal | {folder} | seed {seed}",
            dataset="CIFAR-10",
            device=str(device),
        )
        if board
        else None
    )
    cfg = RunningConfig(
        generations=int(hp["generations"]),
        epochs=int(hp["epochs"]),
        device=device,
        lr_scheduler=LearningRateScheduler(
            ScheduleMode.PROGRESSIVE_PARABOLIC, alpha=float(hp["lr_alpha"])
        ),
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

    try:
        gm, summary = train_generations(
            gm, train_loader, val_loader, cfg, sim_train_loader=sim_train, sim_val_loader=sim_val
        )
    except Exception as exc:
        draw_filtered_fx_graph(gm, str(run_dir / "fx_graph_error_simplified"), fmt="pdf")
        draw_torch_fx_graph(gm, str(run_dir / "fx_graph_error"), fmt="pdf")
        logger.error("Error in train_generations (%s seed %s): %s", folder, seed, exc)
        raise

    _draw_graphs(run_dir, int(summary["generation"][-1]), gm)

    history = {key: summary[key] for key in METRIC_KEYS}
    torch.save(history, run_dir / HISTORY_FILENAME)
    for key in METRIC_KEYS:
        _plot_metric(history[key], key, run_dir / f"{key}.png")

    params_after = GraphStructureQuery.get_amount_of_parameters(gm)
    logger.info(
        "Done %s seed %s val_acc=%.4f params %s -> %s changed=%s",
        folder, seed, max(summary["val_acc"]), params_before, params_after,
        params_after != params_before,
    )


def _assert_cuda_ready(train_device: torch.device) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("train_cifar10 requires CUDA; torch.cuda.is_available() is False")
    cap = torch.cuda.get_device_capability(0)
    logger.info(
        "Training device: %s (%s, sm_%d%d, torch %s)",
        train_device, torch.cuda.get_device_name(0), cap[0], cap[1], torch.__version__,
    )
    try:
        torch.nn.Conv2d(3, 8, 3).to(train_device)(
            torch.zeros(1, *CIFAR_INPUT_SHAPE, device=train_device)
        )
    except RuntimeError as exc:
        if "no kernel image" in str(exc).lower():
            arch = getattr(torch.cuda, "get_arch_list", lambda: [])()
            raise RuntimeError(
                f"PyTorch {torch.__version__} has no CUDA kernels for {torch.cuda.get_device_name(0)} "
                f"(sm_{cap[0]}{cap[1]}). Supported arches: {arch or 'unknown'}. "
                "RTX 50-series often needs cu128 wheels: "
                "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128"
            ) from exc
        raise


if __name__ == "__main__":
    args = _parse_cli()
    device = torch.device("cuda")
    _assert_cuda_ready(device)
    data = Cifar10Data(DATA_DIR)
    data.prepare()
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    runs_done = 0
    for hp in (dict(zip(METAPARAM_KEYS, combo)) for combo in itertools.product(*METAPARAM_LISTS)):
        folder = build_hyperparameter_folder_name(hp)
        for seed in GRID_REPEAT_SEEDS:
            run_dir = run_dir_for_seed(RUNS_DIR, folder, seed)
            if run_dir.exists():
                logger.info("Skipping existing %s seed %s", folder, seed)
                continue
            _run_once(hp, seed=seed, device=device, board=args.board, data=data)
            runs_done += 1
    print(f"Finished {runs_done} run(s) under {RUNS_DIR}" if runs_done else "No pending runs.")
