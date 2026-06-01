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

from growingnn.core.config import RunningConfig
import growingnn.simulation.simulation_algorithms.random_alg as random_alg
from tests.regression.regression_utils import clear_regression_folder, parse_regression_cli


def _running_config():
    x = torch.randn(24, 3, 32, 32)
    y = torch.randint(0, 2, (24,))
    train = DataLoader(TensorDataset(x[:18], y[:18]), batch_size=6)
    val = DataLoader(TensorDataset(x[18:], y[18:]), batch_size=6)
    cfg = RunningConfig(
        generations=1,
        epochs=1,
        criterion=nn.CrossEntropyLoss(),
    )
    cfg.set_simulation_loaders(train, val)
    return cfg


if __name__ == "__main__":
    args = parse_regression_cli()
    torch.manual_seed(0)
    gm = fx.symbolic_trace(resnet18(weights=None, num_classes=2))
    action, depth, rollouts = asyncio.run(random_alg.get_action(gm, _running_config()))
    assert action is not None
    assert depth == 0
    assert rollouts == 0
    if not args.save_output:
        clear_regression_folder()
