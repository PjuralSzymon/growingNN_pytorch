"""CIFAR-10 growingNN run on a tiny two-conv + linear MLP (no ResNet)."""

from __future__ import annotations

import argparse
import itertools
import math
import shutil
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
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

from growingnn.board import ExperimentBoard
from growingnn.core.config import RunningConfig
from growingnn.core.logger import logger
import growingnn.simulation.simulation_algorithms.montecarlo_alg as montecarlo_alg
from growingnn.simulation.score_functions.simulation_score import SimulationScore
from growingnn.training.stoppers import AccuracyStopper
from growingnn.simulation.simulation_scheduler import SchedulerMode, SimulationScheduler
from growingnn.training.lr_scheduler import LearningRateScheduler, ScheduleMode
from growingnn.training.trainer import train_generations
from growingnn.utils.fx import GraphStructureQuery
from growingnn.utils.fx_graph_drawer import draw_filtered_fx_graph, draw_torch_fx_graph

# --- Metaparameter grid (one value per list => original single-run behavior) ---
# ~24 configs x 3 seeds = 72 runs, ~30-44 h on 8 GB GPU
GENERATIONS = [20]
EPOCHS = [30]
BATCH_SIZE = [64]
LR_ALPHA = [0.01]
SIMULATION_TIME = [1000.0]
SIMULATION_EPOCHS = [15]
SIMULATION_SET_SIZE = [2000]
TARGET_ACCURACY = [0.9] # ?
SCORE_WEIGHT_ACC = [1.0] # ?
SCORE_WEIGHT_COUNTW = [0.2, 0.25] # ?
TRAIN_AUGMENTATION = [True]
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
    "train_augmentation",
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
    TRAIN_AUGMENTATION,
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


@dataclass(frozen=True)
class RunResult:
    combo: dict[str, object]
    config_slug: str
    seed: int
    run_dir: Path
    best_val_acc: float
    final_val_acc: float
    params_before: int
    params_after: int
    architecture_changed: bool


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


def _combo_slug(combo: dict[str, object]) -> str:
    aug = "aug" if combo["train_augmentation"] else "noaug"
    return (
        f"g{combo['generations']}_ep{combo['epochs']}_bs{combo['batch_size']}"
        f"_lr{combo['lr_alpha']}_simt{combo['simulation_time']}_sime{combo['simulation_epochs']}"
        f"_simsz{combo['simulation_set_size']}_tgt{combo['target_accuracy']}"
        f"_wacc{combo['score_weight_acc']}_wcw{combo['score_weight_countw']}_{aug}"
        f"_ch{combo['model_channels']}_hd{combo['model_hidden_dim']}"
    )


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


def _train_transform(augment: bool):
    if augment:
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ]
        )
    return transforms.ToTensor()


def _val_transform(augment: bool):
    if augment:
        return transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
        )
    return transforms.ToTensor()


def _loaders(batch_size: int, augment: bool):
    logger.info("Loading CIFAR-10, batch_size %s augment %s", batch_size, augment)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train = datasets.CIFAR10(
        str(DATA_DIR), train=True, download=True, transform=_train_transform(augment)
    )
    val = datasets.CIFAR10(
        str(DATA_DIR), train=False, download=True, transform=_val_transform(augment)
    )
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size)
    logger.info("Loaded CIFAR-10: %s train, %s val samples", len(train), len(val))
    return train_loader, val_loader


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
    config_slug = _combo_slug(combo)
    run_dir.mkdir(parents=True, exist_ok=True)
    _set_seed(seed)
    logger.info("Run %s seed %s -> %s", config_slug, seed, run_dir)

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
            experiment_name=f"CIFAR-10 minimal | {config_slug} | seed {seed}",
            dataset="CIFAR-10",
            device=str(train_device),
        )
        if enable_board
        else None
    )
    cfg = _build_running_config(
        combo, train_device=train_device, board=board, enable_board=enable_board
    )
    train_loader, val_loader = _loaders(int(combo["batch_size"]), bool(combo["train_augmentation"]))

    try:
        gm, summary = train_generations(gm, train_loader, val_loader, cfg)
    except Exception as exc:
        draw_filtered_fx_graph(gm, str(run_dir / "fx_graph_error_simplified"), fmt="pdf")
        draw_torch_fx_graph(gm, str(run_dir / "fx_graph_error"), fmt="pdf")
        logger.error("Error in train_generations (%s seed %s): %s", config_slug, seed, exc)
        raise

    _draw_generation_graphs(run_dir, int(summary["generation"][-1]), gm)
    params_after = GraphStructureQuery.get_amount_of_parameters(gm)
    architecture_changed = params_after != params_before
    logger.info(
        "Run %s seed %s params before %s after %s changed %s",
        config_slug,
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
        combo=combo,
        config_slug=config_slug,
        seed=seed,
        run_dir=run_dir,
        best_val_acc=best_val_acc,
        final_val_acc=final_val_acc,
        params_before=params_before,
        params_after=params_after,
        architecture_changed=architecture_changed,
    )


def _format_combo(combo: dict[str, object]) -> str:
    return ", ".join(f"{key}={combo[key]}" for key in METAPARAM_KEYS)


def _load_existing_run_result(
    combo: dict[str, object], *, seed: int, run_dir: Path
) -> RunResult | None:
    history_path = run_dir / "train_cifar10_history.pt"
    if not history_path.is_file():
        return None
    step_history = torch.load(history_path, map_location="cpu", weights_only=False)
    val_acc = step_history["val_acc"]
    param_count = step_history["param_count"]
    params_before = int(param_count[0])
    params_after = int(param_count[-1])
    return RunResult(
        combo=combo,
        config_slug=_combo_slug(combo),
        seed=seed,
        run_dir=run_dir,
        best_val_acc=max(val_acc),
        final_val_acc=val_acc[-1],
        params_before=params_before,
        params_after=params_after,
        architecture_changed=params_after != params_before,
    )


def _write_grid_summary(results: list[RunResult], path: Path) -> None:
    by_config: dict[str, list[RunResult]] = defaultdict(list)
    for result in results:
        by_config[result.config_slug].append(result)

    config_stats: list[tuple[float, float, str, dict[str, object], list[RunResult]]] = []
    for slug, runs in by_config.items():
        accs = [run.best_val_acc for run in runs]
        mean_acc = statistics.mean(accs)
        std_acc = statistics.pstdev(accs) if len(accs) > 1 else 0.0
        config_stats.append((mean_acc, std_acc, slug, runs[0].combo, runs))

    config_stats.sort(key=lambda item: item[0], reverse=True)
    best_mean, best_std, best_slug, best_combo, best_runs = config_stats[0]

    lines = [
        "GrowingNN CIFAR-10 grid search summary",
        "=" * 72,
        f"Total runs: {len(results)} ({len(by_config)} configs x {len(best_runs)} seeds)",
        "",
        "Configs ranked by mean best validation accuracy:",
    ]
    for rank, (mean_acc, std_acc, slug, combo, runs) in enumerate(config_stats, start=1):
        seeds = ", ".join(str(run.seed) for run in runs)
        acc_list = ", ".join(f"{run.best_val_acc:.4f}" for run in runs)
        lines.append(
            f"{rank:>2}. {slug} | mean={mean_acc:.4f} std={std_acc:.4f} | seeds=[{seeds}] acc=[{acc_list}]"
        )
        lines.append(f"    {_format_combo(combo)}")

    lines.extend(
        [
            "",
            "Best configuration (by mean best val_acc):",
            f"  slug: {best_slug}",
            f"  mean best val_acc: {best_mean:.4f} (std={best_std:.4f})",
            f"  {_format_combo(best_combo)}",
            "",
            "Parameter sensitivity (mean best val_acc per value):",
        ]
    )

    param_spread: list[tuple[str, float, object, object]] = []
    for key in METAPARAM_KEYS:
        grouped: dict[object, list[float]] = defaultdict(list)
        for result in results:
            grouped[result.combo[key]].append(result.best_val_acc)
        lines.append(f"{key}:")
        value_stats = []
        for value, accs in sorted(grouped.items(), key=lambda item: str(item[0])):
            mean_acc = statistics.mean(accs)
            value_stats.append((value, mean_acc))
            lines.append(f"  {value}: mean={mean_acc:.4f} (n={len(accs)})")
        if len(value_stats) > 1:
            best_value, best_value_acc = max(value_stats, key=lambda item: item[1])
            worst_value, worst_value_acc = min(value_stats, key=lambda item: item[1])
            spread = best_value_acc - worst_value_acc
            param_spread.append((key, spread, best_value, worst_value))
        lines.append("")

    lines.append("Suggested tuning priority (largest val_acc spread across tested values):")
    for key, spread, best_value, worst_value in sorted(param_spread, key=lambda item: item[1], reverse=True):
        lines.append(
            f"  {key}: spread={spread:.4f} (best={best_value}, worst={worst_value})"
        )

    lines.extend(
        [
            "",
            "Algorithm notes:",
            "  - Training loop: gradient descent per generation, then MCTS architecture mutation when scheduler allows.",
            "  - Simulation score balances accuracy (weight_acc) vs parameter count (weight_countW); higher weight_acc favors val accuracy.",
            "  - simulation_time / simulation_epochs control MCTS budget; simulation_set_size caps rollout data.",
            "  - generations x epochs set total training budget; target_accuracy can stop early.",
            "  - lr_alpha scales the progressive-parabolic LR schedule; batch_size and augmentation affect SGD stability.",
            "  - model_channels / model_hidden_dim set initial capacity before architecture search grows or shrinks the graph.",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote grid summary to %s", path)


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
        slug = _combo_slug(combo)
        for seed in seeds:
            run_dir = RUNS_DIR / slug / f"seed_{seed}"
            if run_dir.exists():
                result = _load_existing_run_result(combo, seed=seed, run_dir=run_dir)
                if result is None:
                    logger.info("Skipping incomplete run %s seed %s (no history)", slug, seed)
                    continue
                logger.info("Skipping completed run %s seed %s -> %s", slug, seed, run_dir)
            else:
                result = _run_training(
                    combo,
                    seed=seed,
                    run_dir=run_dir,
                    train_device=train_device,
                    enable_board=args.board,
                )
            results.append(result)
            _write_grid_summary(results, SUMMARY_PATH)
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
        baseline = torch.load(HISTORY_PATH, map_location="cpu", weights_only=False)
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
