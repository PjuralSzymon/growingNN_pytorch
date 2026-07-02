"""CIFAR-10 growingNN run on a tiny two-conv + linear MLP (no ResNet)."""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.fx as fx
import torch.nn as nn
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
TARGET_ACCURACY = [0.99, 1.0]
SCORE_WEIGHT_ACC = [1.0, 0.5]  # ?
SCORE_WEIGHT_COUNTW = [0.2, 0.1]  # ?
AUGMENTATION_FACTOR = [0.75, 1.0]  # 0=none, 1=maximum diversity/strength
MODEL_CHANNELS = [32]
MODEL_HIDDEN_DIM = [1024, 2048]
GRID_REPEAT_SEEDS = [30]

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
    "augmentation_factor",
    "model_channels",
    "model_hidden_dim",
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
    AUGMENTATION_FACTOR,
    MODEL_CHANNELS,
    MODEL_HIDDEN_DIM,
)

OUT_DIR = _EXPERIMENT_DIR / "output" / "train_cifar10"
DATA_DIR = OUT_DIR / "data"
RUNS_DIR = OUT_DIR / "runs"
SUMMARY_PATH = OUT_DIR / "grid_search_summary.txt"
NUM_CLASSES = 10
METRIC_KEYS = ("train_loss", "train_acc", "val_loss", "val_acc", "lr", "param_count")
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


class MinimalCifarNet(nn.Module):
    """Two conv layers, one linear hidden, one linear head."""

    def __init__(self, num_classes: int = NUM_CLASSES, channels: int = 8, hidden_dim: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(3, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.hidden = nn.Linear(channels, hidden_dim)
        self.output = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.hidden(x)
        return self.output(x)


class Cifar10Data:
    """CIFAR-10 transforms and DataLoaders for training, validation, and simulation."""

    def __init__(self, data_dir: Path, *, num_workers: int = DATALOADER_NUM_WORKERS) -> None:
        self._data_dir = data_dir
        self._num_workers = num_workers

    @staticmethod
    def clamp_augmentation_factor(augmentation_factor: float) -> float:
        return max(0.0, min(1.0, float(augmentation_factor)))

    @staticmethod
    def eval_transform() -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ]
        )

    @classmethod
    def train_transform(cls, augmentation_factor: float) -> transforms.Compose:
        factor = cls.clamp_augmentation_factor(augmentation_factor)
        if factor <= 0.0:
            return cls.eval_transform()

        steps: list[transforms.Transform] = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        if factor < 0.35:
            pass
        elif factor < 0.70:
            steps.append(transforms.TrivialAugmentWide())
        else:
            steps.append(transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10))

        steps.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ]
        )
        if factor >= 0.85:
            steps.append(
                transforms.RandomErasing(
                    p=0.25,
                    scale=(0.02, 0.20),
                    ratio=(0.3, 3.3),
                    value="random",
                )
            )
        return transforms.Compose(steps)

    def loaders(self, batch_size: int, augmentation_factor: float):
        factor = self.clamp_augmentation_factor(augmentation_factor)
        logger.info(
            "Loading CIFAR-10, batch_size %s augmentation_factor %s simulation_augment False",
            batch_size,
            factor,
        )
        self._data_dir.mkdir(parents=True, exist_ok=True)
        eval_transform = self.eval_transform()
        train_transform = self.train_transform(factor)
        train = datasets.CIFAR10(
            str(self._data_dir), train=True, download=True, transform=train_transform
        )
        train_clean = datasets.CIFAR10(
            str(self._data_dir), train=True, download=True, transform=eval_transform
        )
        val = datasets.CIFAR10(
            str(self._data_dir), train=False, download=True, transform=eval_transform
        )
        loader_kwargs = {"batch_size": batch_size, "num_workers": self._num_workers}
        train_loader = torch.utils.data.DataLoader(train, shuffle=True, **loader_kwargs)
        val_loader = torch.utils.data.DataLoader(val, **loader_kwargs)
        clean_train_loader = torch.utils.data.DataLoader(train_clean, shuffle=False, **loader_kwargs)
        logger.info(
            "Loaded CIFAR-10: %s train, %s val; simulation subset uses non-augmented train images",
            len(train),
            len(val),
        )
        return train_loader, val_loader, clean_train_loader


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
            int(hyperparameters["batch_size"]), float(hyperparameters["augmentation_factor"])
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
    ) -> nn.Module:
        model = MinimalCifarNet(num_classes=num_classes, channels=channels, hidden_dim=hidden_dim)
        logger.info(
            "Built MinimalCifarNet: conv1 3->%s conv2 %s->%s -> pool -> linear %s -> %s",
            model.conv1.out_channels,
            model.conv2.in_channels,
            model.conv2.out_channels,
            model.hidden.out_features,
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
        torch.nn.Conv2d(3, 8, 3).to(train_device)(torch.zeros(1, 3, 32, 32, device=train_device))
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


def _eval_transform() -> transforms.Compose:
    return Cifar10Data.eval_transform()


def _train_transform(augmentation_factor: float) -> transforms.Compose:
    return Cifar10Data.train_transform(augmentation_factor)


if __name__ == "__main__":
    args = _parse_cli()
    train_device = torch.device("cuda")
    _assert_cuda_ready(train_device)
    Cifar10Experiment(args, train_device).run()
