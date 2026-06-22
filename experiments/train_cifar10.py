"""CIFAR-10 growingNN run on a tiny two-conv + linear MLP (no ResNet)."""

from __future__ import annotations

import argparse
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

from growingnn.core.config import RunningConfig
from growingnn.core.logger import logger
import growingnn.simulation.simulation_algorithms.montecarlo_alg as montecarlo_alg
from growingnn.simulation.score_functions.simulation_score import SimulationScore
from growingnn.training.stoppers import AccuracyStopper, StopperMode, TrainingStopper
from growingnn.simulation.simulation_scheduler import SchedulerMode, SimulationScheduler
from growingnn.training.lr_scheduler import LearningRateScheduler, ScheduleMode
from growingnn.training.trainer import train_generations
from growingnn.utils.fx import GraphStructureQuery
from growingnn.utils.fx_graph_drawer import draw_filtered_fx_graph, draw_torch_fx_graph

OUT_DIR = _EXPERIMENT_DIR / "output" / "train_cifar10"
DATA_DIR = OUT_DIR / "data"
HISTORY_PATH = OUT_DIR / "train_cifar10_history.pt"
NUM_CLASSES = 10
METRIC_KEYS = ("train_loss", "train_acc", "val_loss", "val_acc", "lr", "param_count")


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
    ns = parser.parse_args(argv)
    ns.save_output = ns.save_output == "true"
    return ns


def _clear_output_dir() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def _build_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    model = MinimalCifarNet(num_classes=num_classes, channels=32, hidden_dim=256)
    logger.info(
        "Built MinimalCifarNet: conv1 3->%s conv2 %s->%s -> pool -> linear %s -> %s",
        model.conv1.out_channels,
        model.conv2.in_channels,
        model.conv2.out_channels,
        model.hidden.out_features,
        num_classes,
    )
    return model


def _loaders(batch_size: int = 64):
    logger.info("Loading CIFAR-10, batch_size %s", batch_size)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    transform = transforms.ToTensor()
    train = datasets.CIFAR10(str(DATA_DIR), train=True, download=True, transform=transform)
    val = datasets.CIFAR10(str(DATA_DIR), train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, num_workers=0)
    logger.info("Loaded CIFAR-10: %s train, %s val samples", len(train), len(val))
    return train_loader, val_loader


def _draw_generation_graphs(generation: int, gm: fx.GraphModule) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    draw_filtered_fx_graph(gm, str(OUT_DIR / f"fx_graph_simplified{generation}"), fmt="pdf")
    draw_torch_fx_graph(gm, str(OUT_DIR / f"fx_graph{generation}"), fmt="pdf")


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


def _load_step_history(path: Path) -> dict[str, list[float]]:
    data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict in {path}, got {type(data).__name__}")
    return data


if __name__ == "__main__":
    args = _parse_cli()
    torch.manual_seed(0)

    gm = fx.symbolic_trace(_build_model())
    params_before = GraphStructureQuery.get_amount_of_parameters(gm)
    _draw_generation_graphs(0, gm)

    cfg = RunningConfig(
        generations=20,
        epochs=20,
        lr_scheduler=LearningRateScheduler(ScheduleMode.PROGRESSIVE_PARABOLIC, alpha=0.01),
        print_every=1,
        simulation_alg=montecarlo_alg,
        simulation_scheduler=SimulationScheduler(
            SchedulerMode.ALWAYS, simulation_time=600.0, simulation_epochs=10
        ),
        stopper = AccuracyStopper(target_accuracy=0.9),
        simulation_score=SimulationScore(weight_acc=1.0, weight_countW=0.25),
        simulation_set_size=64,
        criterion=nn.CrossEntropyLoss(),
        quiet=False,
    )
    try:
        gm, summary = train_generations(gm, *_loaders(), cfg)   
    except Exception as e:
        draw_filtered_fx_graph(gm, str(OUT_DIR / "fx_graph_error_simplified"), fmt="pdf")
        draw_torch_fx_graph(gm, str(OUT_DIR / "fx_graph_error"), fmt="pdf")
        logger.error("Error in train_generations: %s", e)
        raise
    _draw_generation_graphs(int(summary["generation"][-1]), gm)

    params_after = GraphStructureQuery.get_amount_of_parameters(gm)
    logger.info("Parameters before %s after %s", params_before, params_after)
    assert params_after != params_before, "architecture search did not change the model"

    step_history = {key: summary[key] for key in METRIC_KEYS}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for key in METRIC_KEYS:
        _plot_metric(step_history[key], key, OUT_DIR / f"{key}.png")

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
