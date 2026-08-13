"""Unit tests for learning-based simulation scores."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from growingnn.core.config import RunningConfig
from growingnn.simulation.score_functions.score_by_learning import (
    AccuracyMetric,
    parse_accuracy_metric,
    score_acc,
    score_loss,
)
from growingnn.simulation.score_functions.simulation_score import SimulationScore


def _running_config(accuracy_metric: AccuracyMetric | str = AccuracyMetric.VAL_ACC) -> RunningConfig:
    x = torch.randn(16, 4)
    y = torch.randint(0, 2, (16,))
    train = DataLoader(TensorDataset(x[:12], y[:12]), batch_size=4)
    val = DataLoader(TensorDataset(x[12:], y[12:]), batch_size=4)
    cfg = RunningConfig(generations=1, epochs=1)
    cfg.set_simulation_loaders(train, val)
    cfg.simulation_score = SimulationScore(accuracy_metric=accuracy_metric)
    return cfg


def test_parse_accuracy_metric_accepts_aliases():
    """
    parse_accuracy_metric should normalize common train/val aliases.
    """
    # Arrange / Act / Assert
    assert parse_accuracy_metric("val_acc") is AccuracyMetric.VAL_ACC
    assert parse_accuracy_metric("train") is AccuracyMetric.TRAIN_ACC
    assert parse_accuracy_metric(None) is AccuracyMetric.VAL_ACC


def test_parse_accuracy_metric_rejects_unknown_value():
    """
    parse_accuracy_metric should raise ValueError for unsupported metric names.
    """
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="Unknown accuracy metric"):
        parse_accuracy_metric("f1")


def test_score_acc_uses_validation_accuracy_by_default():
    """
    score_acc should return the last val_acc when SimulationScore uses val_acc.
    """
    # Arrange
    cfg = _running_config(AccuracyMetric.VAL_ACC)
    model = nn.Linear(2, 1)

    # Act
    with patch(
        "growingnn.simulation.score_functions.score_by_learning.run_simulation_scoring_gradient_descent",
        return_value=(None, {"val_acc": [0.2, 0.55], "train_acc": [0.9]}),
    ):
        result = score_acc(model, cfg)

    # Assert
    assert result == 0.55


def test_score_acc_uses_training_accuracy_when_configured():
    """
    score_acc should return the last train_acc when SimulationScore uses train_acc.
    """
    # Arrange
    cfg = _running_config(AccuracyMetric.TRAIN_ACC)
    model = nn.Linear(2, 1)

    # Act
    with patch(
        "growingnn.simulation.score_functions.score_by_learning.run_simulation_scoring_gradient_descent",
        return_value=(None, {"val_acc": [0.55], "train_acc": [0.2, 0.81]}),
    ):
        result = score_acc(model, cfg)

    # Assert
    assert result == 0.81


def test_score_loss_prefers_lower_validation_loss():
    """
    score_loss should return a higher score for lower val_loss (min(1 / (max(loss, 1e-8) + 1), 1)).
    """

    # Arrange
    cfg = _running_config()
    model = nn.Linear(2, 1)

    # Act
    with patch(
        "growingnn.simulation.score_functions.score_by_learning.run_simulation_scoring_gradient_descent",
        side_effect=[(None, {"val_loss": [0.1]}), (None, {"val_loss": [0.9]})],
    ):
        low_loss_score = score_loss(model, cfg)
        high_loss_score = score_loss(model, cfg)

    # Assert
    assert low_loss_score > high_loss_score
    assert low_loss_score == 1.0 / (0.1 + 1)
    assert high_loss_score == 1.0 / (0.9 + 1)


def test_score_loss_uses_training_loss_when_configured():
    """
    score_loss should grade from train_loss when accuracy_metric is train_acc.
    """
    # Arrange
    cfg = _running_config(AccuracyMetric.TRAIN_ACC)
    model = nn.Linear(2, 1)

    # Act
    with patch(
        "growingnn.simulation.score_functions.score_by_learning.run_simulation_scoring_gradient_descent",
        return_value=(None, {"train_loss": [0.25], "val_loss": [9.0]}),
    ):
        result = score_loss(model, cfg)

    # Assert
    assert result == 1.0 / (0.25 + 1)


def test_score_loss_reward_stays_in_open_interval_zero_one():
    """
    score_loss should map non-negative losses into (0, 1] without saturating below loss 1.
    """

    # Arrange
    cfg = _running_config()
    model = nn.Linear(2, 1)

    # Act
    with patch(
        "growingnn.simulation.score_functions.score_by_learning.run_simulation_scoring_gradient_descent",
        side_effect=[(None, {"val_loss": [0.01]}), (None, {"val_loss": [0.99]})],
    ):
        very_low = score_loss(model, cfg)
        moderate = score_loss(model, cfg)

    # Assert
    assert 0.0 < very_low <= 1.0
    assert 0.0 < moderate <= 1.0
    assert very_low != moderate
