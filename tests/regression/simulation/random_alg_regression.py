import sys
from pathlib import Path

import asyncio
import torch
import torch.fx as fx
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet18

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from growingnn.simulation.context import SimulationContext
import growingnn.simulation.simulation_algorithms.random_alg as random_alg
from growingnn.training.lr_scheduler import LearningRateScheduler, ScheduleMode
from tests.regression.regression_utils import clear_regression_folder, parse_regression_cli


def _ctx():
    x = torch.randn(24, 3, 32, 32)
    y = torch.randint(0, 2, (24,))
    train = DataLoader(TensorDataset(x[:18], y[:18]), batch_size=6)
    val = DataLoader(TensorDataset(x[18:], y[18:]), batch_size=6)
    return SimulationContext(
        train_loader=train,
        val_loader=val,
        criterion=nn.CrossEntropyLoss(),
        lr_scheduler=LearningRateScheduler(ScheduleMode.CONSTANT, alpha=0.01),
        epochs=1,
    )


if __name__ == "__main__":
    args = parse_regression_cli()
    torch.manual_seed(0)
    gm = fx.symbolic_trace(resnet18(weights=None, num_classes=2))
    action, depth, rollouts = asyncio.run(random_alg.get_action(gm, 1.0, _ctx(), None))
    assert action is not None
    assert depth == 0
    assert rollouts == 0
    if not args.save_output:
        clear_regression_folder()
