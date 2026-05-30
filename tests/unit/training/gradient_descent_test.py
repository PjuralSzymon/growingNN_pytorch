"""Unit tests for ``growingnn.training.gradient_descent``."""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from growingnn.training.gradient_descent import gradient_descent
from growingnn.training.lr_scheduler import LearningRateScheduler, ScheduleMode
from growingnn.training.stoppers import StopperMode, TrainingStopper


def test_gradient_descent_records_history_for_each_epoch():
    """
    gradient_descent should train the model and append metrics for every epoch pass.
    """
    # Arrange
    torch.manual_seed(0)
    inputs = torch.randn(32, 4)
    targets = torch.randint(0, 2, (32,))
    train_loader = DataLoader(TensorDataset(inputs, targets), batch_size=8)
    val_loader = DataLoader(TensorDataset(inputs, targets), batch_size=8)
    model = nn.Linear(4, 2)
    criterion = nn.CrossEntropyLoss()
    lr_scheduler = LearningRateScheduler(ScheduleMode.CONSTANT, alpha=0.1)
    stopper = TrainingStopper(StopperMode.EMPTY)
    epochs = 2

    # Act
    trained_model, history = gradient_descent(
        model,
        epochs,
        train_loader,
        val_loader,
        criterion,
        lr_scheduler,
        stopper=stopper,
        quiet=True,
    )

    # Assert
    assert trained_model is model
    assert len(history["train_loss"]) == epochs + 1
    assert len(history["train_acc"]) == epochs + 1
    assert len(history["val_loss"]) == epochs + 1
    assert len(history["val_acc"]) == epochs + 1
    assert history["lr"] == [0.1, 0.1, 0.1]


def test_gradient_descent_stops_when_stopper_triggers():
    """
    gradient_descent should break early when the stopper returns True on a print step.
    """
    # Arrange
    torch.manual_seed(0)
    inputs = torch.randn(16, 4)
    targets = torch.randint(0, 2, (16,))
    train_loader = DataLoader(TensorDataset(inputs, targets), batch_size=4)
    val_loader = DataLoader(TensorDataset(inputs, targets), batch_size=4)
    model = nn.Linear(4, 2)
    criterion = nn.CrossEntropyLoss()
    lr_scheduler = LearningRateScheduler(ScheduleMode.CONSTANT, alpha=0.1)
    stopper = TrainingStopper(StopperMode.ACCURACY, target_accuracy=0.0)
    epochs = 20

    # Act
    _, history = gradient_descent(
        model,
        epochs,
        train_loader,
        val_loader,
        criterion,
        lr_scheduler,
        stopper=stopper,
        quiet=False,
        print_every=1,
    )

    # Assert
    assert len(history["train_loss"]) < epochs + 1


def test_gradient_descent_accepts_custom_optimizer():
    """
    gradient_descent should train with a caller-provided optimizer instance.
    """
    # Arrange
    torch.manual_seed(0)
    inputs = torch.randn(32, 4)
    targets = torch.randint(0, 2, (32,))
    train_loader = DataLoader(TensorDataset(inputs, targets), batch_size=8)
    val_loader = DataLoader(TensorDataset(inputs, targets), batch_size=8)
    model = nn.Linear(4, 2)
    criterion = nn.CrossEntropyLoss()
    lr_scheduler = LearningRateScheduler(ScheduleMode.CONSTANT, alpha=0.05)
    stopper = TrainingStopper(StopperMode.EMPTY)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Act
    _, history = gradient_descent(
        model,
        1,
        train_loader,
        val_loader,
        criterion,
        lr_scheduler,
        stopper=stopper,
        optimizer=optimizer,
        quiet=True,
    )

    # Assert
    assert len(history["train_loss"]) == 2
    assert history["lr"] == [0.05, 0.05]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
