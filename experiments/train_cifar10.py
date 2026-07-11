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
from growingnn.training.stoppers import AccuracyStopper
from growingnn.simulation.simulation_scheduler import SchedulerMode, SimulationScheduler
from growingnn.training.lr_scheduler import LearningRateScheduler, ScheduleMode
from growingnn.simulation.simulation_set import sample_loaders
from growingnn.training.trainer import train_generations
from growingnn.utils.fx import GraphStructureQuery
from growingnn.utils.fx_graph_drawer import draw_filtered_fx_graph, draw_torch_fx_graph

from createsummary import (
    HISTORY_FILENAME,
    RunResult,
    build_hyperparameter_folder_name,
    collect_run_results,
    run_dir_for_seed,
    write_grid_summary,
)

# --- Metaparameter grid (one value per list => original single-run behavior) ---
# ~24 configs x 3 seeds = 72 runs, ~30-44 h on 8 GB GPU
GENERATIONS = [10, 20]
EPOCHS = [30]
BATCH_SIZE = [64]
LR_ALPHA = [0.01]
SIMULATION_TIME = [500.0]
SIMULATION_EPOCHS = [15]
SIMULATION_SET_SIZE = [2000]
TARGET_ACCURACY = [0.99]
SCORE_WEIGHT_ACC = [1.0]  # ?
SCORE_WEIGHT_COUNTW = [0.2]  # ?
MODEL_CHANNELS = [32]
MODEL_HIDDEN_DIM = [256]
MODEL_NUM_BLOCKS = [1]
GRID_REPEAT_SEEDS = [110]

METAPARAM_KEYS = (
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
    "model_hidden_dim",
    "model_num_blocks",
)
METAPARAM_LISTS = (
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
    MODEL_HIDDEN_DIM,
    MODEL_NUM_BLOCKS,
)

OUT_DIR = _EXPERIMENT_DIR / "output" / "train_cifar10"
DATA_DIR = _EXPERIMENT_DIR / "data" / "cifar10"
RUNS_DIR = OUT_DIR / "runs"
SUMMARY_PATH = OUT_DIR / "grid_search_summary.txt"
NUM_CLASSES = 10
METRIC_KEYS = ("train_loss", "train_acc", "val_loss", "val_acc", "lr", "param_count")
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CIFAR_INPUT_SHAPE = (3, 32, 32)


class Cifar10Data:
    """CIFAR-10 loaders; datasets and DataLoaders are built once per process."""

    def __init__(self, data_dir: Path, *, num_workers: int = DATALOADER_NUM_WORKERS) -> None:
        self._data_dir = data_dir
        self._num_workers = num_workers
        self._datasets: tuple[datasets.CIFAR10, datasets.CIFAR10, datasets.CIFAR10] | None = None
        self._loader_cache: dict[int, tuple[DataLoader, DataLoader, DataLoader]] = {}

    @staticmethod
    def eval_transform() -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ]
        )

    @classmethod
    def train_transform(cls) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ]
        )

    def prepare(self) -> None:
        """Load torchvision datasets once (safe to call before the grid loop)."""
        if self._datasets is not None:
            return
        self._data_dir.mkdir(parents=True, exist_ok=True)
        download = not (self._data_dir / "cifar-10-batches-py").is_dir()
        root = str(self._data_dir)
        eval_transform = self.eval_transform()
        train_transform = self.train_transform()
        self._datasets = (
            datasets.CIFAR10(root, train=True, download=download, transform=train_transform),
            datasets.CIFAR10(root, train=True, download=download, transform=eval_transform),
            datasets.CIFAR10(root, train=False, download=download, transform=eval_transform),
        )
        train, _, val = self._datasets
        logger.info("Loaded CIFAR-10: %s train, %s val", len(train), len(val))

    def loaders(self, batch_size: int) -> tuple[DataLoader, DataLoader, DataLoader]:
        self.prepare()
        if batch_size in self._loader_cache:
            return self._loader_cache[batch_size]
        assert self._datasets is not None
        train, train_clean, val = self._datasets
        loader_kwargs: dict[str, object] = {
            "batch_size": batch_size,
            "num_workers": self._num_workers,
        }
        if self._num_workers > 0:
            loader_kwargs["persistent_workers"] = True
        pin_memory = torch.cuda.is_available()
        loaders = (
            DataLoader(train, shuffle=True, pin_memory=pin_memory, **loader_kwargs),
            DataLoader(val, pin_memory=pin_memory, **loader_kwargs),
            DataLoader(train_clean, shuffle=False, pin_memory=pin_memory, **loader_kwargs),
        )
        self._loader_cache[batch_size] = loaders
        return loaders


class MinimalBasicBlock(nn.Module):
    """Single ResNet basic block (3x3 convs + optional 1x1 shortcut)."""

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
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
        self.conv1 = nn.Conv2d(3, channels, kernel_size=3, stride=1, padding=1, bias=False)
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
        x = torch.flatten(x, 1)
        return self.linear(x)


class Cifar10TrainingRun:
    """Train one hyperparameter configuration and seed; save artifacts under run_dir."""

    def __init__(
        self,
        *,
        data: Cifar10Data,
        train_device: torch.device,
        enable_board: bool,
    ) -> None:
        self._data = data
        self._train_device = train_device
        self._enable_board = enable_board

    def run(self, hyperparameters: dict[str, object], *, seed: int, run_dir: Path) -> RunResult:
        hyperparameter_folder_name = build_hyperparameter_folder_name(hyperparameters)
        if not run_dir.exists():
            run_dir.mkdir(parents=True, exist_ok=True)
        self._set_seed(seed)
        logger.info("Run %s seed %s -> %s", hyperparameter_folder_name, seed, run_dir)

        gm = fx.symbolic_trace(
            self._build_model(
                channels=int(hyperparameters["model_channels"]),
                hidden_dim=int(hyperparameters["model_hidden_dim"]),
                num_blocks=int(hyperparameters["model_num_blocks"]),
            )
        )
        params_before = GraphStructureQuery.get_amount_of_parameters(gm)
        self._draw_generation_graphs(run_dir, 0, gm)

        board = (
            ExperimentBoard(
                run_dir / "board",
                experiment_name=f"CIFAR-10 minimal | {hyperparameter_folder_name} | seed {seed}",
                dataset="CIFAR-10",
                device=str(self._train_device),
            )
            if self._enable_board
            else None
        )
        cfg = self._build_running_config(
            hyperparameters, board=board, enable_board=self._enable_board
        )
        train_loader, val_loader, clean_train_loader = self._data.loaders(
            int(hyperparameters["batch_size"])
        )
        sim_train_loader, sim_val_loader = sample_loaders(
            clean_train_loader,
            val_loader,
            int(hyperparameters["simulation_set_size"]),
            seed=seed,
        )

        try:
            gm, summary = train_generations(
                gm,
                train_loader,
                val_loader,
                cfg,
                sim_train_loader=sim_train_loader,
                sim_val_loader=sim_val_loader,
            )
        except Exception as exc:
            draw_filtered_fx_graph(gm, str(run_dir / "fx_graph_error_simplified"), fmt="pdf")
            draw_torch_fx_graph(gm, str(run_dir / "fx_graph_error"), fmt="pdf")
            logger.error(
                "Error in train_generations (%s seed %s): %s",
                hyperparameter_folder_name,
                seed,
                exc,
            )
            raise

        self._draw_generation_graphs(run_dir, int(summary["generation"][-1]), gm)
        params_after = GraphStructureQuery.get_amount_of_parameters(gm)
        architecture_changed = params_after != params_before
        logger.info(
            "Run %s seed %s params before %s after %s changed %s",
            hyperparameter_folder_name,
            seed,
            params_before,
            params_after,
            architecture_changed,
        )

        step_history = {key: summary[key] for key in METRIC_KEYS}
        for key in METRIC_KEYS:
            self._plot_metric(step_history[key], key, run_dir / f"{key}.png")
        torch.save(step_history, run_dir / HISTORY_FILENAME)

        return RunResult(
            hyperparameters=hyperparameters,
            hyperparameter_folder_name=hyperparameter_folder_name,
            seed=seed,
            run_dir=run_dir,
            best_val_acc=max(summary["val_acc"]),
            final_val_acc=summary["val_acc"][-1],
            params_before=params_before,
            params_after=params_after,
            architecture_changed=architecture_changed,
        )

    @staticmethod
    def _set_seed(seed: int) -> None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _build_model(
        num_classes: int = NUM_CLASSES,
        *,
        channels: int = 32,
        hidden_dim: int = 256,
        num_blocks: int = 1,
    ) -> nn.Module:
        model = MinimalCifarNet(
            num_classes=num_classes,
            channels=channels,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
        )
        widths = [channels] + (
            [channels, hidden_dim] if num_blocks == 2 else [hidden_dim]
        )
        logger.info(
            "Built MinimalCifarNet: stem 3->%s blocks=%s widths %s linear %s -> %s",
            channels,
            num_blocks,
            widths,
            hidden_dim,
            num_classes,
        )
        return model

    def _build_running_config(
        self,
        hyperparameters: dict[str, object],
        *,
        board: ExperimentBoard | None,
        enable_board: bool,
    ) -> RunningConfig:
        return RunningConfig(
            generations=int(hyperparameters["generations"]),
            epochs=int(hyperparameters["epochs"]),
            device=self._train_device,
            lr_scheduler=LearningRateScheduler(
                ScheduleMode.PROGRESSIVE_PARABOLIC, alpha=float(hyperparameters["lr_alpha"])
            ),
            print_every=1,
            simulation_alg=montecarlo_alg,
            simulation_scheduler=SimulationScheduler(
                SchedulerMode.ALWAYS,
                simulation_time=float(hyperparameters["simulation_time"]),
                simulation_epochs=int(hyperparameters["simulation_epochs"]),
            ),
            stopper=AccuracyStopper(target_accuracy=float(hyperparameters["target_accuracy"])),
            simulation_score=SimulationScore(
                weight_acc=float(hyperparameters["score_weight_acc"]),
                weight_countW=float(hyperparameters["score_weight_countw"]),
            ),
            simulation_set_size=int(hyperparameters["simulation_set_size"]),
            criterion=nn.CrossEntropyLoss(),
            quiet=False,
            enable_experiment_board=enable_board,
            experiment_board=board,
        )

    @staticmethod
    def _draw_generation_graphs(out_dir: Path, generation: int, gm: fx.GraphModule) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        draw_filtered_fx_graph(gm, str(out_dir / f"fx_graph_simplified{generation}"), fmt="pdf")
        draw_torch_fx_graph(gm, str(out_dir / f"fx_graph{generation}"), fmt="pdf")

    @staticmethod
    def _plot_metric(values: list[float], name: str, save_path: Path) -> None:
        steps = range(1, len(values) + 1)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(steps, values)
        ax.set_xlabel("step")
        ax.set_ylabel(name)
        fig.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)


class Cifar10Experiment:
    """Run the hyperparameter grid sequentially; skip run dirs that already exist."""

    def __init__(self, args: argparse.Namespace, train_device: torch.device) -> None:
        self._args = args
        self._train_device = train_device
        self._data = Cifar10Data(DATA_DIR)
        self._trainer = Cifar10TrainingRun(
            data=self._data, train_device=train_device, enable_board=args.board
        )

    def run(self) -> None:
        self._data.prepare()
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        runs_done = 0
        for hyperparameters in self._iter_hyperparameter_sets():
            hyperparameter_folder_name = build_hyperparameter_folder_name(hyperparameters)
            for seed in GRID_REPEAT_SEEDS:
                run_dir = run_dir_for_seed(RUNS_DIR, hyperparameter_folder_name, seed)
                if run_dir.exists():
                    logger.info("Skipping existing %s seed %s", hyperparameter_folder_name, seed)
                    continue
                logger.info(
                    "Starting experiment %s seed %s -> %s",
                    hyperparameter_folder_name,
                    seed,
                    run_dir,
                )
                self._trainer.run(hyperparameters, seed=seed, run_dir=run_dir)
                runs_done += 1
        if runs_done:
            print(f"Finished {runs_done} run(s). Summary: {SUMMARY_PATH}")
        else:
            print("No pending runs.")

    @staticmethod
    def _iter_hyperparameter_sets() -> list[dict[str, object]]:
        return [dict(zip(METAPARAM_KEYS, combo)) for combo in itertools.product(*METAPARAM_LISTS)]


def _parse_cli(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="train_cifar10 minimal growingNN experiment")
    parser.add_argument(
        "--board",
        choices=("true", "false"),
        default="true",
        help="Write GrowingNN Board artifacts under each run's board/ folder (default: true)",
    )
    ns = parser.parse_args(argv)
    ns.board = ns.board == "true"
    return ns


def _assert_cuda_ready(train_device: torch.device) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("train_cifar10 requires CUDA; torch.cuda.is_available() is False")
    cap = torch.cuda.get_device_capability(0)
    logger.info(
        "Training device: %s (%s, sm_%d%d, torch %s)",
        train_device,
        torch.cuda.get_device_name(0),
        cap[0],
        cap[1],
        torch.__version__,
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


def _install_cifar_shape_probe() -> None:
    """Patch FX ShapeProp fallback probe to match CIFAR-10 (3x32x32), not ImageNet 224."""
    from growingnn.utils.fx.graph_analysis import LayerShapeAnalyser

    spatial = CIFAR_INPUT_SHAPE[1]

    @staticmethod
    def _cifar_default_example_input(gm: fx.GraphModule) -> torch.Tensor | None:
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
        return torch.randn(1, *CIFAR_INPUT_SHAPE, device=device, dtype=dtype)

    LayerShapeAnalyser.default_example_input = _cifar_default_example_input


if __name__ == "__main__":
    args = _parse_cli()
    train_device = torch.device("cuda")
    _assert_cuda_ready(train_device)
    _install_cifar_shape_probe()
    Cifar10Experiment(args, train_device).run()
