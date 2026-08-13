from typing import List
import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from growingnn.actions.add_seq_linear_layer import AddSeqLinearLayer
from growingnn.training.gradient_descent import gradient_descent
from growingnn.training.lr_scheduler_action import ActionLearningRateScheduler, LearningRateScheduler, ScheduleMode
from growingnn.training.stoppers import StopperMode, TrainingStopper
from growingnn.utils.fx_graph_drawer import draw_filtered_fx_graph, draw_torch_fx_graph
from growingnn.core.traced_model import TracedModel
from tests.regression.regression_utils import (
    FOLDER_NAME,
    clear_regression_folder,
    parse_regression_cli,
    regression_cifar_dir,
)
OUT_DIR = FOLDER_NAME + "/training"
DATA_DIR = regression_cifar_dir()
HISTORY_PATH = OUT_DIR + "/simple_gradient_descent_history.pt"
NUM_CLASSES = 10
METRIC_KEYS = ("train_loss", "train_acc", "val_loss", "val_acc", "lr")


def _loaders(seed: int = 0, train_n: int = 128, val_n: int = 32, batch_size: int = 8):
    os.makedirs(DATA_DIR, exist_ok=True)
    transform = transforms.ToTensor()
    has_cifar = os.path.isdir(os.path.join(DATA_DIR, "cifar-10-batches-py"))
    train_full = datasets.CIFAR10(DATA_DIR, train=True, download=not has_cifar, transform=transform)
    val_full = datasets.CIFAR10(DATA_DIR, train=False, download=not has_cifar, transform=transform)

    generator = torch.Generator().manual_seed(seed)
    train_idx = torch.randperm(len(train_full), generator=generator)[:train_n].tolist()
    val_idx = torch.randperm(len(val_full), generator=generator)[:val_n].tolist()

    train = DataLoader(Subset(train_full, train_idx), batch_size=batch_size, shuffle=True)
    val = DataLoader(Subset(val_full, val_idx), batch_size=batch_size)
    return train, val


def _combine_histories(histories: list[dict[str, list[float]]]) -> dict[str, list[float]]:
    combined: dict[str, list[float]] = {key: [] for key in histories[0]}
    for history in histories:
        for key in combined:
            combined[key].extend(history[key])
    return combined


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


def _plot_training_histories(combined: dict[str, list[float]], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for key in METRIC_KEYS:
        _plot_metric(combined[key], key, out_dir + f"/{key}.png")


if __name__ == "__main__":
    args = parse_regression_cli()
    torch.manual_seed(0)
    model = resnet18(weights=None, num_classes=NUM_CLASSES)
    gm = torch.fx.symbolic_trace(model)
    train_loader, val_loader = _loaders()
    rng = random.Random(42)
    previous_acc = 0.0
    all_histories: list[dict[str, list[float]]] = []
    for id in range(5):
        _, history = gradient_descent(
            gm,
            5,
            train_loader,
            val_loader,
            nn.CrossEntropyLoss(),
            ActionLearningRateScheduler(ScheduleMode.CONSTANT, alpha=0.01),
            TrainingStopper(StopperMode.EMPTY),
            quiet=False,
            print_every=1,
        )
        actions: List[AddSeqLinearLayer] = AddSeqLinearLayer.generate_all_actions(TracedModel.create(gm, (1, 3, 32, 32)))
        idx = rng.randrange(len(actions))
        draw_filtered_fx_graph(gm, FOLDER_NAME + "/" + "fx_graph_simplified" + str(id), fmt="pdf")
        draw_torch_fx_graph(gm, FOLDER_NAME + "/" + "fx_graph" + str(id), fmt="pdf")
        actions[idx].execute(TracedModel.create(gm, (1, 3, 32, 32)))
        all_histories.append(history)
        print(f"generation: {id} action executed: {actions[idx]}")
        assert history["train_acc"][0] >= previous_acc * 0.9
        assert history["train_acc"][-1] > 0.5
        previous_acc = history["train_acc"][-1]

    combined_history = _combine_histories(all_histories)
    _plot_training_histories(combined_history, OUT_DIR)

    if args.save_output or not os.path.exists(HISTORY_PATH):
        os.makedirs(OUT_DIR, exist_ok=True)
        torch.save(combined_history, HISTORY_PATH)
        if not args.save_output:
            print(f"Baseline missing; wrote {HISTORY_PATH}. Re-run with --save-output true to refresh.")
    else:
        baseline = torch.load(HISTORY_PATH, map_location="cpu", weights_only=False)
        for key in combined_history:
            assert combined_history[key] == baseline[key]

    if not args.save_output:
        clear_regression_folder()
