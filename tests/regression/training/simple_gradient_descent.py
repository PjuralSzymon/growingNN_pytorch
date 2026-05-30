import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet18

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from growingnn.training.gradient_descent import gradient_descent
from growingnn.training.lr_scheduler import LearningRateScheduler, ScheduleMode
from growingnn.training.stoppers import StopperMode, TrainingStopper
from tests.regression.regression_utils import FOLDER_NAME, clear_regression_folder, parse_regression_cli

OUT_DIR = FOLDER_NAME + "/training"
HISTORY_PATH = OUT_DIR + "/simple_gradient_descent_history.pt"


def _loaders(seed: int = 0, n: int = 64, batch_size: int = 8):
    torch.manual_seed(seed)
    x = torch.randn(n, 3, 32, 32)
    y = torch.randint(0, 2, (n,))
    train = DataLoader(TensorDataset(x[:48], y[:48]), batch_size=batch_size)
    val = DataLoader(TensorDataset(x[48:], y[48:]), batch_size=batch_size)
    return train, val


if __name__ == "__main__":
    args = parse_regression_cli()
    torch.manual_seed(0)
    model = resnet18(weights=None, num_classes=2)
    train_loader, val_loader = _loaders()
    _, history = gradient_descent(
        model,
        2,
        train_loader,
        val_loader,
        nn.CrossEntropyLoss(),
        LearningRateScheduler(ScheduleMode.CONSTANT, alpha=0.01),
        TrainingStopper(StopperMode.EMPTY),
        quiet=True,
    )

    assert history["train_loss"][-1] < history["train_loss"][0]

    if args.save_output:
        os.makedirs(OUT_DIR, exist_ok=True)
        torch.save(history, HISTORY_PATH)
    else:
        baseline = torch.load(HISTORY_PATH, map_location="cpu", weights_only=False)
        for key in history:
            assert history[key] == baseline[key]

    if not args.save_output:
        clear_regression_folder()
