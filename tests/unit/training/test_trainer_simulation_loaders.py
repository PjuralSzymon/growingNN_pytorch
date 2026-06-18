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
from growingnn.simulation.simulation_scheduler import SchedulerMode, SimulationScheduler
from growingnn.training.lr_scheduler import LearningRateScheduler, ScheduleMode
from growingnn.training.stoppers import StopperMode, TrainingStopper
from growingnn.training.trainer import train_generations


class _TinyNet(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=(2, 3))


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
        lr_scheduler=LearningRateScheduler(ScheduleMode.CONSTANT, alpha=0.01),
        stopper=TrainingStopper(StopperMode.EMPTY),
        simulation_scheduler=SimulationScheduler(SchedulerMode.NEVER),
        simulation_set_size=8,
        criterion=nn.CrossEntropyLoss(),
        quiet=True,
    )

    # Act
    with patch("growingnn.training.trainer.sample_loaders") as mock_sample:
        train_generations(
            gm,
            train_loader,
            val_loader,
            cfg,
            sim_train_loader=sim_train_loader,
            sim_val_loader=sim_val_loader,
        )

    # Assert
    mock_sample.assert_not_called()
    assert cfg.sim_train_loader is sim_train_loader
    assert cfg.sim_val_loader is sim_val_loader
