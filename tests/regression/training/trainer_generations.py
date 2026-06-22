import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.fx as fx
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from growingnn.core.config import RunningConfig
from growingnn.simulation.score_functions.simulation_score import SimulationScore
import growingnn.simulation.simulation_algorithms.montecarlo_alg as montecarlo_alg
from growingnn.simulation.simulation_scheduler import SchedulerMode, SimulationScheduler
from growingnn.training.lr_scheduler import LearningRateScheduler, ScheduleMode
from growingnn.training.trainer import train_generations
from growingnn.utils.fx import GraphStructureQuery
from growingnn.utils.fx_graph_drawer import draw_filtered_fx_graph, draw_torch_fx_graph
from growingnn.core.logger import logger

from tests.regression.regression_utils import (
    FOLDER_NAME,
    clear_regression_folder,
    parse_regression_cli,
)

OUT_DIR = FOLDER_NAME + "/training"
DATA_DIR = OUT_DIR + "/data"
HISTORY_PATH = OUT_DIR + "/trainer_generations_history.pt"
WEIGHTS_PATH = OUT_DIR + "/resnet18_cifar10_baseline.pt"
WEIGHTS_URL = (
    "https://huggingface.co/Phoenix21/resnet18-cifar10-baseline/resolve/main/"
    "resnet18_cifar10_baseline.pth"
)
NUM_CLASSES = 10
METRIC_KEYS = ("train_loss", "train_acc", "val_loss", "val_acc", "lr", "param_count")


def _load_cifar10_resnet18(num_classes: int = NUM_CLASSES) -> nn.Module:
    model = resnet18(weights=None, num_classes=num_classes)
    if not os.path.exists(WEIGHTS_PATH):
        logger.info("Downloading CIFAR-10 ResNet18 weights from %s", WEIGHTS_URL)
        os.makedirs(OUT_DIR, exist_ok=True)
        state_dict = torch.hub.load_state_dict_from_url(
            WEIGHTS_URL, progress=True, map_location="cpu"
        )
        torch.save(state_dict, WEIGHTS_PATH)
    else:
        state_dict = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    logger.info("Loaded CIFAR-10 pretrained ResNet18 weights")
    return model


def _loaders(batch_size: int = 128):
    logger.info(f"Loading full CIFAR10, batch_size {batch_size}")
    os.makedirs(DATA_DIR, exist_ok=True)
    transform = transforms.ToTensor()
    train = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=transform)
    val = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=transform)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size)
    logger.info(f"Loaded CIFAR10: {len(train)} train, {len(val)} val samples")
    return train_loader, val_loader


def _draw_generation_graphs(generation: int, gm: fx.GraphModule) -> None:
    draw_filtered_fx_graph(gm, FOLDER_NAME + "/" + "fx_graph_simplified" + str(generation), fmt="pdf")
    draw_torch_fx_graph(gm, FOLDER_NAME + "/" + "fx_graph" + str(generation), fmt="pdf")


def _plot_metric(values: list[float], name: str, save_path: str) -> None:
    steps = range(1, len(values) + 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps, values)
    ax.set_xlabel("step")
    ax.set_ylabel(name)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    args = parse_regression_cli()
    torch.manual_seed(0)
    gm = fx.symbolic_trace(_load_cifar10_resnet18())
    params_before = GraphStructureQuery.get_amount_of_parameters(gm)
    _draw_generation_graphs(0, gm)
    cfg = RunningConfig(
        generations=10,
        epochs=10,
        lr_scheduler=LearningRateScheduler(ScheduleMode.PROGRESSIVE, alpha=0.01, steepness=0.5),
        print_every=1,
        simulation_alg=montecarlo_alg,
        simulation_scheduler=SimulationScheduler(
            SchedulerMode.ALWAYS, simulation_time=120.0, simulation_epochs=10
        ),
        simulation_score=SimulationScore(weight_acc=0.0, weight_countW=1.0),
        simulation_set_size=100,
        criterion=nn.CrossEntropyLoss(),
        quiet=False,
    )
    gm, summary = train_generations(gm, *_loaders(), cfg)
    _draw_generation_graphs(summary["generation"][-1], gm)
    assert GraphStructureQuery.get_amount_of_parameters(gm) != params_before

    step_history = {key: summary[key] for key in METRIC_KEYS}
    os.makedirs(OUT_DIR, exist_ok=True)
    for key in METRIC_KEYS:
        _plot_metric(step_history[key], key, OUT_DIR + f"/{key}.png")

    if args.save_output or not os.path.exists(HISTORY_PATH):
        os.makedirs(OUT_DIR, exist_ok=True)
        torch.save(step_history, HISTORY_PATH)
        if not args.save_output:
            print(f"Baseline missing; wrote {HISTORY_PATH}. Re-run with --save-output true to refresh.")
    else:
        baseline = torch.load(HISTORY_PATH, map_location="cpu", weights_only=False)
        for key in step_history:
            assert step_history[key] == baseline[key]

    if not args.save_output:
        clear_regression_folder()
