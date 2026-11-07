"""Unit tests for simulation scoring optimizer settings."""

from unittest.mock import patch

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from growingnn.core.config import RunningConfig
from growingnn.simulation.score_functions.simulation_training import (
    run_simulation_scoring_gradient_descent,
)
from growingnn.simulation.simulation_schedulers import NeverSimulationScheduler
from growingnn.training.lr_scheduler_action import ActionLearningRateScheduler, ScheduleMode


def test_simulation_scoring_forwards_momentum_and_weight_decay():
    """
    Simulation scoring should train with the same momentum and weight decay as the run.
    """
    # Arrange
    torch.manual_seed(0)
    x = torch.randn(8, 4)
    y = torch.randint(0, 2, (8,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    model = nn.Linear(4, 2)
    cfg = RunningConfig(
        generations=1,
        epochs=1,
        lr_scheduler=ActionLearningRateScheduler(ScheduleMode.CONSTANT, alpha=0.01),
        simulation_scheduler=NeverSimulationScheduler(simulation_epochs=1),
        criterion=nn.CrossEntropyLoss(),
        momentum=0.9,
        weight_decay=5e-4,
        quiet=True,
    )
    cfg.set_simulation_loaders(loader, loader)
    captured: dict[str, float] = {}

    def _fake_gradient_descent(*_args, **kwargs):
        captured["momentum"] = kwargs["momentum"]
        captured["weight_decay"] = kwargs["weight_decay"]
        return model, {}

    # Act
    with patch(
        "growingnn.simulation.score_functions.simulation_training.gradient_descent",
        _fake_gradient_descent,
    ):
        run_simulation_scoring_gradient_descent(model, cfg)

    # Assert
    assert captured["momentum"] == 0.9
    assert captured["weight_decay"] == 5e-4
