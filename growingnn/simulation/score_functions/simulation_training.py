"""Shared simulation GD used by score terms."""

from __future__ import annotations

import torch.fx as fx
import torch.nn as nn

from growingnn.core.config import RunningConfig
from growingnn.training.gradient_descent import gradient_descent


def run_simulation_scoring_gradient_descent(
    model: nn.Module | fx.GraphModule,
    running_config: RunningConfig,
) -> tuple[nn.Module | fx.GraphModule, dict[str, list[float]]]:
    """
    Run short GD for scoring.

    Each scoring run trains with a restarted copy of the main learning-rate schedule, so every
    candidate in every generation is trained the way the main run starts, and scoring never
    moves the main run's schedule.
    """
    return gradient_descent(
        model,
        running_config.simulation_scheduler.simulation_epochs,
        running_config.sim_train_loader,
        running_config.sim_val_loader,
        running_config.criterion,
        running_config.lr_scheduler.restarted(),
        quiet=True,
        momentum=running_config.momentum,
        weight_decay=running_config.weight_decay,
        device=running_config.device,
        max_grad_norm=running_config.max_grad_norm,
    )
