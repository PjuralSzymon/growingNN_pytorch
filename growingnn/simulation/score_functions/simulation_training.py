"""Shared simulation GD used by score terms."""

from __future__ import annotations

import torch.fx as fx
import torch.nn as nn

from growingnn.core.config import RunningConfig
from growingnn.training.gradient_descent import gradient_descent
from growingnn.training.lr_scheduler_global import (
    freeze_global_learning_rate_progress_if_supported,
)


def run_simulation_scoring_gradient_descent(
    model: nn.Module | fx.GraphModule,
    running_config: RunningConfig,
) -> tuple[nn.Module | fx.GraphModule, dict[str, list[float]]]:
    """
    Run short GD for scoring with global LR progress frozen when composed.

    Freezing keeps a shared ComposedLearningRateScheduler from advancing
    global_epoch during MCTS / simulation rollouts.
    """
    with freeze_global_learning_rate_progress_if_supported(running_config.lr_scheduler):
        return gradient_descent(
            model,
            running_config.simulation_scheduler.simulation_epochs,
            running_config.sim_train_loader,
            running_config.sim_val_loader,
            running_config.criterion,
            running_config.lr_scheduler,
            quiet=True,
            device=running_config.device,
        )
