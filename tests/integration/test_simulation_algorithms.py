"""Integration tests for architecture simulation algorithms."""

import asyncio
import sys
from pathlib import Path

import pytest
import torch
import torch.fx as fx
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet18

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import growingnn.core.config

growingnn.core.config.ENABLE_LOGGING = False

from growingnn.core.config import RunningConfig
from growingnn.simulation.score_functions.simulation_score import SimulationScore
import growingnn.simulation.simulation_algorithms.greedy_alg as greedy_alg
import growingnn.simulation.simulation_algorithms.random_alg as random_alg
from growingnn.utils.fx import GraphStructureQuery


def _running_config(*, simulation_time: float = 1.0, simulation_score=None):
    x = torch.randn(32, 3, 32, 32)
    y = torch.randint(0, 2, (32,))
    train = DataLoader(TensorDataset(x[:24], y[:24]), batch_size=8, shuffle=True)
    val = DataLoader(TensorDataset(x[24:], y[24:]), batch_size=8)
    cfg = RunningConfig(
        generations=1,
        epochs=1,
        criterion=nn.CrossEntropyLoss(),
    )
    cfg.set_simulation_loaders(train, val)
    cfg.simulation_scheduler.simulation_time = simulation_time
    cfg.simulation_score = simulation_score
    return cfg


def test_random_alg_returns_executable_action():
    """
    random_alg should pick an action that can mutate a traced ResNet-18.
    """
    # Arrange
    gm = fx.symbolic_trace(resnet18(weights=None, num_classes=2))
    params_before = GraphStructureQuery.get_amount_of_parameters(gm)
    cfg = _running_config()

    # Act
    action, _, _ = asyncio.run(random_alg.get_action(gm, cfg))

    # Assert
    assert action is not None
    action.execute(gm)
    assert GraphStructureQuery.get_amount_of_parameters(gm) != params_before


def test_greedy_alg_returns_action_within_time_budget():
    """
    greedy_alg should finish within the allotted time and return an action.
    """
    # Arrange
    gm = fx.symbolic_trace(resnet18(weights=None, num_classes=2))
    cfg = _running_config(
        simulation_time=2.0,
        simulation_score=SimulationScore(weight_acc=1.0, weight_countW=0.0),
    )

    # Act
    action, _, rollouts = asyncio.run(greedy_alg.get_action(gm, cfg))

    # Assert
    assert action is not None
    assert rollouts >= 1
