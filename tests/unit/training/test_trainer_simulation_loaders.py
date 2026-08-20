"""Unit tests for optional pre-built simulation loaders in train_generations."""

import sys
from pathlib import Path
from unittest.mock import patch

import torch
import torch.fx as fx
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import growingnn.core.config

growingnn.core.config.ENABLE_LOGGING = False

from growingnn.core.config import RunningConfig
from growingnn.simulation.simulation_schedulers import NeverSimulationScheduler
from growingnn.training.lr_scheduler_action import ActionLearningRateScheduler, LearningRateScheduler, ScheduleMode
from growingnn.training.stoppers import StopperMode, TrainingStopper
from growingnn.training.trainer import train_generations


class _TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x.mean(dim=(2, 3)))


def test_train_generations_uses_prebuilt_simulation_loaders():
    """
    train_generations should keep caller-provided sim loaders instead of sampling train_loader.
    """
    # Arrange
    torch.manual_seed(0)
    gm = fx.symbolic_trace(_TinyNet())
    x = torch.randn(16, 3, 32, 32)
    y = torch.randint(0, 2, (16,))
    train_loader = DataLoader(TensorDataset(x[:12], y[:12]), batch_size=4, shuffle=True)
    val_loader = DataLoader(TensorDataset(x[12:], y[12:]), batch_size=4)
    sim_train_loader = DataLoader(TensorDataset(x[:8], y[:8]), batch_size=4)
    sim_val_loader = DataLoader(TensorDataset(x[8:12], y[8:12]), batch_size=4)
    cfg = RunningConfig(
        generations=1,
        epochs=1,
        lr_scheduler=ActionLearningRateScheduler(ScheduleMode.CONSTANT, alpha=0.01),
        stopper=TrainingStopper(StopperMode.EMPTY),
        simulation_scheduler=NeverSimulationScheduler(),
        simulation_set_size=8,
        criterion=nn.CrossEntropyLoss(),
        quiet=True,
    )

    # Act
    with patch.object(cfg.simulation_set_generator, "generate") as mock_generate:
        train_generations(
            gm,
            train_loader,
            val_loader,
            cfg,
            sim_train_loader=sim_train_loader,
            sim_val_loader=sim_val_loader,
        )

    # Assert
    mock_generate.assert_not_called()
    assert cfg.sim_train_loader is sim_train_loader
    assert cfg.sim_val_loader is sim_val_loader


def test_train_generations_builds_sim_loaders_when_missing():
    """
    When sim loaders are not passed, generate should be called with train_loader, val_loader, and size.
    """

    # Arrange
    torch.manual_seed(0)
    gm = fx.symbolic_trace(_TinyNet())
    x = torch.randn(16, 3, 32, 32)
    y = torch.randint(0, 2, (16,))
    train_loader = DataLoader(TensorDataset(x[:12], y[:12]), batch_size=4)
    val_loader = DataLoader(TensorDataset(x[12:], y[12:]), batch_size=4)
    dummy_sim_train = DataLoader(TensorDataset(x[:8], y[:8]), batch_size=4)
    dummy_sim_val = DataLoader(TensorDataset(x[8:12], y[8:12]), batch_size=4)
    cfg = RunningConfig(
        generations=1,
        epochs=1,
        lr_scheduler=ActionLearningRateScheduler(ScheduleMode.CONSTANT, alpha=0.01),
        stopper=TrainingStopper(StopperMode.EMPTY),
        simulation_scheduler=NeverSimulationScheduler(),
        simulation_set_size=8,
        criterion=nn.CrossEntropyLoss(),
        quiet=True,
    )

    # Act
    with patch.object(
        cfg.simulation_set_generator,
        "generate",
        return_value=(dummy_sim_train, dummy_sim_val),
    ) as mock_generate:
        train_generations(gm, train_loader, val_loader, cfg)

    # Assert
    mock_generate.assert_called_once_with(train_loader, val_loader, 8)
    assert cfg.sim_train_loader is dummy_sim_train
    assert cfg.sim_val_loader is dummy_sim_val
