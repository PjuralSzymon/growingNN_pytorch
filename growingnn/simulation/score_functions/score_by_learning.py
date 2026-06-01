"""Learning-based simulation score terms."""

from __future__ import annotations

import torch.fx as fx
import torch.nn as nn

from growingnn.core.config import RunningConfig
from growingnn.training.gradient_descent import gradient_descent


def score_acc(
    model: nn.Module | fx.GraphModule,
    running_config: RunningConfig,
) -> float:
    _, history = gradient_descent(
        model,
        running_config.simulation_scheduler.simulation_epochs,
        running_config.sim_train_loader,
        running_config.sim_val_loader,
        running_config.criterion,
        running_config.lr_scheduler,
        quiet=True,
    )
    return float(history["val_acc"][-1])  # TODO: originally train acc


def score_loss(
    model: nn.Module | fx.GraphModule,
    running_config: RunningConfig,
) -> float:
    _, history = gradient_descent(
        model,
        running_config.simulation_scheduler.simulation_epochs,
        running_config.sim_train_loader,
        running_config.sim_val_loader,
        running_config.criterion,
        running_config.lr_scheduler,
        quiet=True,
    )
    loss = float(history["val_loss"][-1])
    return min(1.0 / (max(loss, 1e-8) + 1), 1.0)  # TODO: originally train loss
