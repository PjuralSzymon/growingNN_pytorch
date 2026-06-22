"""CIFAR-10 growingNN run on a tiny two-conv + linear MLP (no ResNet)."""

from __future__ import annotations

import argparse
import itertools
import math
import shutil
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
from growingnn.core.config import RunningConfig
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
    RunResult,
    build_hyperparameter_folder_name,
    load_run_result_from_dir,
    write_grid_summary,
)

# --- Metaparameter grid (one value per list => original single-run behavior) ---
# ~24 configs x 3 seeds = 72 runs, ~30-44 h on 8 GB GPU
GENERATIONS = [10]
EPOCHS = [30]
BATCH_SIZE = [64]
LR_ALPHA = [0.01]
SIMULATION_TIME = [500.0]
SIMULATION_EPOCHS = [15]
SIMULATION_SET_SIZE = [2000]
TARGET_ACCURACY = [0.99]
SCORE_WEIGHT_ACC = [1.0] # ?
SCORE_WEIGHT_COUNTW = [0.2] # ?
AUGMENTATION_FACTOR = [0.0, 0.2, 0.5, 0.75, 1.0]  # 0=none, 1=maximum diversity/strength
MODEL_CHANNELS = [32]
MODEL_HIDDEN_DIM = [256]
GRID_REPEAT_SEEDS = [0]

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
HISTORY_PATH = OUT_DIR / "train_cifar10_history.pt"
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


def _parse_cli(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="train_cifar10 minimal growingNN experiment")
    parser.add_argument(
        "--save-output",
        "--save_output",
        choices=("true", "false"),
        default="false",
        help="Keep experiments/output/train_cifar10 and refresh baseline (default: false)",
    )
    parser.add_argument(
        "--board",
        choices=("true", "false"),
        default="true",
        help="Write GrowingNN Board artifacts under each run's board/ folder (default: true)",
    )
    ns = parser.parse_args(argv)
    ns.save_output = ns.save_output == "true"
    ns.board = ns.board == "true"
    return ns


def _is_grid_mode() -> bool:
    return math.prod(len(values) for values in METAPARAM_LISTS) > 1


def _iter_combos() -> list[dict[str, object]]:
    return [dict(zip(METAPARAM_KEYS, combo)) for combo in itertools.product(*METAPARAM_LISTS)]


def _clear_output_dir() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def _eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def _clamp_augmentation_factor(augmentation_factor: float) -> float:
    return max(0.0, min(1.0, float(augmentation_factor)))

from torchvision import transforms

def _train_transform(augmentation_factor: float) -> transforms.Compose:
    """
    CIFAR-10 / ResNet training transform.

    Recommended strategy:
    - always use CIFAR baseline: crop + horizontal flip
    - use only ONE strong policy: AutoAugment OR TrivialAugment/RandAugment
    - do not stack TrivialAugment + RandAugment + affine + heavy jitter
    """
    factor = _clamp_augmentation_factor(augmentation_factor)

    if factor <= 0.0:
        return _eval_transform()

    steps: list[transforms.Transform] = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]

    if factor < 0.35:
        # Light, safe baseline.
        pass

    elif factor < 0.70:
        # Strong but simple. Good default if you want little tuning.
        steps.append(transforms.TrivialAugmentWide())

    else:
        # Best CIFAR-10-specific image-level policy.
        steps.append(
            transforms.AutoAugment(
                policy=transforms.AutoAugmentPolicy.CIFAR10
            )
        )

    steps.extend([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    # Optional Cutout-like regularization.
    # Keep it mild; do not use huge erase probability on 32x32 images.
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


def _loaders(batch_size: int, augmentation_factor: float):
    factor = _clamp_augmentation_factor(augmentation_factor)
    logger.info(
        "Loading CIFAR-10, batch_size %s augmentation_factor %s simulation_augment False",
        batch_size,
        factor,
    )
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    eval_transform = _eval_transform()
    train_transform = _train_transform(factor)
    train = datasets.CIFAR10(
        str(DATA_DIR), train=True, download=True, transform=train_transform
    )
    train_clean = datasets.CIFAR10(
        str(DATA_DIR), train=True, download=True, transform=eval_transform
    )
    val = datasets.CIFAR10(
        str(DATA_DIR), train=False, download=True, transform=eval_transform
    )
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size)
    clean_train_loader = torch.utils.data.DataLoader(
        train_clean, batch_size=batch_size, shuffle=False
    )
    logger.info(
        "Loaded CIFAR-10: %s train, %s val; simulation subset uses non-augmented train images",
        len(train),
        len(val),
    )
    return train_loader, val_loader, clean_train_loader


def _draw_generation_graphs(out_dir: Path, generation: int, gm: fx.GraphModule) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    draw_filtered_fx_graph(gm, str(out_dir / f"fx_graph_simplified{generation}"), fmt="pdf")
    draw_torch_fx_graph(gm, str(out_dir / f"fx_graph{generation}"), fmt="pdf")


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


def _build_running_config(
    combo: dict[str, object],
    *,
    train_device: torch.device,
    board: ExperimentBoard | None,
    enable_board: bool,
) -> RunningConfig:
    return RunningConfig(
        generations=int(combo["generations"]),
        epochs=int(combo["epochs"]),
        device=train_device,
        lr_scheduler=LearningRateScheduler(
            ScheduleMode.PROGRESSIVE_PARABOLIC, alpha=float(combo["lr_alpha"])
        ),
        print_every=1,
        simulation_alg=montecarlo_alg,
        simulation_scheduler=SimulationScheduler(
            SchedulerMode.ALWAYS,
            simulation_time=float(combo["simulation_time"]),
            simulation_epochs=int(combo["simulation_epochs"]),
        ),
        stopper=AccuracyStopper(target_accuracy=float(combo["target_accuracy"])),
        simulation_score=SimulationScore(
            weight_acc=float(combo["score_weight_acc"]),
            weight_countW=float(combo["score_weight_countw"]),
        ),
        simulation_set_size=int(combo["simulation_set_size"]),
        criterion=nn.CrossEntropyLoss(),
        quiet=False,
        enable_experiment_board=enable_board,
        experiment_board=board,
    )


def _run_training(
    combo: dict[str, object],
    *,
    seed: int,
    run_dir: Path,
    train_device: torch.device,
    enable_board: bool,
) -> RunResult:
    hyperparameter_folder_name = build_hyperparameter_folder_name(combo)
    run_dir.mkdir(parents=True, exist_ok=True)
    _set_seed(seed)
    logger.info("Run %s seed %s -> %s", hyperparameter_folder_name, seed, run_dir)

    gm = fx.symbolic_trace(
        _build_model(
            channels=int(combo["model_channels"]),
            hidden_dim=int(combo["model_hidden_dim"]),
        )
    )
    params_before = GraphStructureQuery.get_amount_of_parameters(gm)
    _draw_generation_graphs(run_dir, 0, gm)

    board = (
        ExperimentBoard(
            run_dir / "board",
            experiment_name=f"CIFAR-10 minimal | {hyperparameter_folder_name} | seed {seed}",
            dataset="CIFAR-10",
            device=str(train_device),
        )
        if enable_board
        else None
    )
    cfg = _build_running_config(
        combo, train_device=train_device, board=board, enable_board=enable_board
    )
    train_loader, val_loader, clean_train_loader = _loaders(
        int(combo["batch_size"]), float(combo["augmentation_factor"])
    )
    sim_train_loader, sim_val_loader = sample_loaders(
        clean_train_loader,
        val_loader,
        int(combo["simulation_set_size"]),
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
        logger.error("Error in train_generations (%s seed %s): %s", hyperparameter_folder_name, seed, exc)
        raise

    _draw_generation_graphs(run_dir, int(summary["generation"][-1]), gm)
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
        _plot_metric(step_history[key], key, run_dir / f"{key}.png")
    torch.save(step_history, run_dir / "train_cifar10_history.pt")

    best_val_acc = max(summary["val_acc"])
    final_val_acc = summary["val_acc"][-1]
    return RunResult(
        hyperparameters=combo,
        hyperparameter_folder_name=hyperparameter_folder_name,
        seed=seed,
        run_dir=run_dir,
        best_val_acc=best_val_acc,
        final_val_acc=final_val_acc,
        params_before=params_before,
        params_after=params_after,
        architecture_changed=architecture_changed,
    )


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


def _run_grid(args: argparse.Namespace, train_device: torch.device) -> None:
    combos = _iter_combos()
    seeds = GRID_REPEAT_SEEDS
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results: list[RunResult] = []
    for combo in combos:
        hyperparameter_folder_name = build_hyperparameter_folder_name(combo)
        for seed in seeds:
            run_dir = RUNS_DIR / hyperparameter_folder_name / f"seed_{seed}"
            if run_dir.exists():
                result = load_run_result_from_dir(
                    run_dir,
                    hyperparameters=combo,
                    hyperparameter_folder_name=hyperparameter_folder_name,
                    seed=seed,
                )
                if result is None:
                    logger.info(
                        "Skipping incomplete run %s seed %s (no history)",
                        hyperparameter_folder_name,
                        seed,
                    )
                    continue
                logger.info(
                    "Skipping completed run %s seed %s -> %s",
                    hyperparameter_folder_name,
                    seed,
                    run_dir,
                )
            else:
                result = _run_training(
                    combo,
                    seed=seed,
                    run_dir=run_dir,
                    train_device=train_device,
                    enable_board=args.board,
                )
            results.append(result)
            write_grid_summary(results, SUMMARY_PATH)
    print(f"Grid search finished. Summary: {SUMMARY_PATH}")


def _run_single(args: argparse.Namespace, train_device: torch.device) -> None:
    combo = _iter_combos()[0]
    run_dir = OUT_DIR
    result = _run_training(
        combo,
        seed=0,
        run_dir=run_dir,
        train_device=train_device,
        enable_board=args.board,
    )
    if not result.architecture_changed:
        raise AssertionError("architecture search did not change the model")

    step_history = torch.load(run_dir / "train_cifar10_history.pt", map_location="cpu", weights_only=False)
    if args.save_output or not HISTORY_PATH.is_file():
        torch.save(step_history, HISTORY_PATH)
        if not args.save_output:
            print(f"Baseline missing; wrote {HISTORY_PATH}. Re-run with --save-output true to refresh.")
    else:
        baseline = _load_step_history(HISTORY_PATH)
        for key in step_history:
            assert step_history[key] == baseline[key]

    if not args.save_output:
        _clear_output_dir()


if __name__ == "__main__":
    args = _parse_cli()
    train_device = torch.device("cuda")
    _assert_cuda_ready(train_device)

    if _is_grid_mode():
        _run_grid(args, train_device)
    else:
        _run_single(args, train_device)
