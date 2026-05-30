"""Unit tests for ``growingnn.training.stoppers``."""

import sys
from pathlib import Path

import pytest
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from growingnn.training.stoppers import (
    AccuracyStopper,
    EmptyStopper,
    ParameterCountStopper,
    StopperMode,
    TrainingStopper,
    _parameter_count,
)


def test_empty_stopper_never_stops():
    """
    EmptyStopper should always return False regardless of metrics.
    """
    # Arrange
    stopper = EmptyStopper()
    model = nn.Linear(4, 2)
    metrics = {"accuracy": 1.0}

    # Act
    result_early = stopper.check(model, epoch=1, metrics=metrics)
    result_late = stopper.check(model, epoch=100, metrics=metrics)

    # Assert
    assert result_early is False
    assert result_late is False


def test_accuracy_stopper_stops_when_target_is_reached():
    """
    AccuracyStopper should stop once reported accuracy meets the target.
    """
    # Arrange
    stopper = AccuracyStopper(target_accuracy=0.9)
    model = nn.Linear(4, 2)

    # Act
    result_below = stopper.check(model, epoch=1, metrics={"accuracy": 0.5})
    result_at_target = stopper.check(model, epoch=2, metrics={"accuracy": 0.9})
    result_above = stopper.check(model, epoch=3, metrics={"accuracy": 0.95})

    # Assert
    assert result_below is False
    assert result_at_target is True
    assert result_above is True


def test_accuracy_stopper_ignores_missing_metrics():
    """
    AccuracyStopper should not stop when accuracy is missing from metrics.
    """
    # Arrange
    stopper = AccuracyStopper(target_accuracy=0.5)
    model = nn.Linear(4, 2)

    # Act
    result_none = stopper.check(model, epoch=1, metrics=None)
    result_empty = stopper.check(model, epoch=2, metrics={})

    # Assert
    assert result_none is False
    assert result_empty is False


def test_parameter_count_stopper_stops_after_parameter_reduction():
    """
    ParameterCountStopper should stop after parameter count drops enough from the initial count.
    """
    # Arrange
    stopper = ParameterCountStopper(decrease_threshold=0.5)
    large_model = nn.Linear(100, 100)
    small_model = nn.Linear(10, 10)

    # Act
    result_init = stopper.check(large_model, epoch=0, metrics=None)
    stopper.reset()
    stopper.check(large_model, epoch=0, metrics=None)
    result_reduced = stopper.check(small_model, epoch=1, metrics=None)

    # Assert
    assert result_init is False
    assert result_reduced is True


def test_parameter_count_tracks_pytorch_module_parameters():
    """
    _parameter_count should match the total numel of model parameters.
    """
    # Arrange
    model = nn.Sequential(nn.Linear(3, 5), nn.Linear(5, 2))
    expected = sum(p.numel() for p in model.parameters())

    # Act
    result = _parameter_count(model)

    # Assert
    assert result == expected


def test_training_stopper_dispatches_from_mode():
    """
    TrainingStopper should delegate to the stopper matching StopperMode.
    """
    # Arrange
    empty = TrainingStopper(StopperMode.EMPTY)
    accuracy = TrainingStopper(StopperMode.ACCURACY, target_accuracy=0.8)
    model = nn.Linear(4, 2)
    metrics = {"accuracy": 0.85}

    # Act
    result_empty = empty.check(model, epoch=1, metrics=metrics)
    result_accuracy = accuracy.check(model, epoch=1, metrics=metrics)

    # Assert
    assert result_empty is False
    assert result_accuracy is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
