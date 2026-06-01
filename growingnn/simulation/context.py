"""Shared context for simulation rollouts and scoring."""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn
from torch.utils.data import DataLoader

from growingnn.core.config import RunningConfig


@dataclass
class SimulationContext:
    train_loader: DataLoader
    val_loader: DataLoader
    criterion: nn.Module
    running_config: RunningConfig
