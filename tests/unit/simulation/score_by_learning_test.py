"""Unit tests for learning-based simulation scores."""

import sys
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from growingnn.core.config import RunningConfig
from growingnn.simulation.score_functions.score_by_learning import score_loss


def _running_config() -> RunningConfig:
    x = torch.randn(16, 4)
    y = torch.randint(0, 2, (16,))
    train = DataLoader(TensorDataset(x[:12], y[:12]), batch_size=4)
    val = DataLoader(TensorDataset(x[12:], y[12:]), batch_size=4)
    cfg = RunningConfig(generations=1, epochs=1)
    cfg.set_simulation_loaders(train, val)
    return cfg


def test_score_loss_prefers_lower_validation_loss():
    """
    score_loss should return a higher score for lower val_loss (min(1 / (max(loss, 1e-8) + 1), 1)).
    """

    # Arrange
    cfg = _running_config()
    model = nn.Linear(2, 1)

    # Act
    with patch(
        "growingnn.simulation.score_functions.score_by_learning.gradient_descent",
        side_effect=[(None, {"val_loss": [0.1]}), (None, {"val_loss": [0.9]})],
    ):
        low_loss_score = score_loss(model, cfg)
        high_loss_score = score_loss(model, cfg)

    # Assert
    assert low_loss_score > high_loss_score
    assert low_loss_score == 1.0 / (0.1 + 1)
    assert high_loss_score == 1.0 / (0.9 + 1)


def test_score_loss_reward_stays_in_open_interval_zero_one():
    """
    score_loss should map non-negative losses into (0, 1] without saturating below loss 1.
    """

    # Arrange
    cfg = _running_config()
    model = nn.Linear(2, 1)

    # Act
    with patch(
        "growingnn.simulation.score_functions.score_by_learning.gradient_descent",
        side_effect=[(None, {"val_loss": [0.01]}), (None, {"val_loss": [0.99]})],
    ):
        very_low = score_loss(model, cfg)
        moderate = score_loss(model, cfg)

    # Assert
    assert 0.0 < very_low <= 1.0
    assert 0.0 < moderate <= 1.0
    assert very_low != moderate
