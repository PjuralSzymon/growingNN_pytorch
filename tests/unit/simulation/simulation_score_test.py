"""Unit tests for simulation scoring."""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from growingnn.simulation.context import SimulationContext
from growingnn.simulation.score_functions.score_efficiency import score_count_weights
from growingnn.simulation.score_functions.simulation_score import SimulationScore
from growingnn.training.lr_scheduler import LearningRateScheduler, ScheduleMode


def _ctx(epochs: int = 1):
    x = torch.randn(16, 4)
    y = torch.randint(0, 2, (16,))
    train = DataLoader(TensorDataset(x[:12], y[:12]), batch_size=4)
    val = DataLoader(TensorDataset(x[12:], y[12:]), batch_size=4)
    return SimulationContext(
        train_loader=train,
        val_loader=val,
        criterion=nn.CrossEntropyLoss(),
        lr_scheduler=LearningRateScheduler(ScheduleMode.CONSTANT, alpha=0.05),
        epochs=epochs,
    )


def test_score_count_weights_prefers_smaller_models():
    """
    score_count_weights should return a higher score for models with fewer parameters.
    """
    # Arrange
    small = nn.Linear(4, 2)
    large = nn.Sequential(nn.Linear(4, 128), nn.Linear(128, 2))
    ctx = _ctx()

    # Act
    small_score = score_count_weights(small, ctx)
    large_score = score_count_weights(large, ctx)

    # Assert
    assert small_score > large_score


def test_simulation_score_returns_weighted_value():
    """
    SimulationScore should combine enabled score terms into one scalar.
    """
    # Arrange
    model = nn.Linear(4, 2)
    score_fn = SimulationScore(weight_acc=0.0, weight_loss=0.0, weight_time=0.0, weight_countW=1.0)
    ctx = _ctx()

    # Act
    result = score_fn.score(model, ctx)

    # Assert
    assert 0.0 < result <= 1.0
