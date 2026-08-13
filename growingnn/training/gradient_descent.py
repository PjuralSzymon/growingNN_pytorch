"""Training loop for growingnn models with pluggable optimizers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import torch
import torch.fx as fx
import torch.nn as nn
from torch.utils.data import DataLoader

from growingnn.training.lr_scheduler_action import MIN_LEARNING_RATE, LearningRateScheduler

if TYPE_CHECKING:
    from growingnn.board.experiment_board import ExperimentBoard
from growingnn.training.stoppers import StopperMode, TrainingStopper

PROGRESS_PRINT_FREQUENCY = 10


def _optimizer_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def _resolve_optimizer(
    parameters: Iterable[torch.nn.Parameter],
    optimizer: torch.optim.Optimizer | None,
    momentum: float,
) -> torch.optim.Optimizer:
    """
    Return the caller optimizer, or build SGD with a throwaway ctor LR.

    Epoch 0 always overwrites LR via alpha_scheduler before any training steps.
    """
    if optimizer is not None:
        return optimizer
    # SGD requires an lr in the constructor; epoch 0 overwrites it before any steps.
    return torch.optim.SGD(
        parameters,
        lr=MIN_LEARNING_RATE,
        momentum=momentum,
        weight_decay=0.0,
    )


def _evaluate(
    model: nn.Module | fx.GraphModule,
    loader: DataLoader[Any],
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs.float())
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(dim=1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    if total == 0:
        return 0.0, 0.0
    return running_loss / total, correct / total


def gradient_descent(
    model: nn.Module | fx.GraphModule,
    epochs: int,
    train_loader: DataLoader[Any],
    val_loader: DataLoader[Any],
    criterion: nn.Module,
    lr_scheduler: LearningRateScheduler,
    stopper: TrainingStopper | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    momentum: float = 0.0,
    device: str | torch.device = "cpu",
    quiet: bool = False,
    print_every: int = PROGRESS_PRINT_FREQUENCY,
    experiment_board: ExperimentBoard | None = None,
    generation: int = 0,
) -> tuple[nn.Module | fx.GraphModule, dict[str, list[float]]]:
    if epochs <= 0:
        raise ValueError("Number of epochs must be positive")
    if lr_scheduler is None:
        raise ValueError("Learning rate scheduler cannot be None")

    stopper = stopper or TrainingStopper(StopperMode.EMPTY)
    device = torch.device(device)
    model = model.to(device)

    optimizer = _resolve_optimizer(model.parameters(), optimizer, momentum)
    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    for epoch in range(epochs):
        scheduled_lr = lr_scheduler.alpha_scheduler(epoch, epochs)
        _set_optimizer_lr(optimizer, scheduled_lr)

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs.float())
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(dim=1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = correct / total if total else 0.0
        val_loss, val_acc = _evaluate(model, val_loader, criterion, device)
        applied_lr = _optimizer_lr(optimizer)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(applied_lr)
        metrics = {"accuracy": train_acc, "val_acc": val_acc, "loss": train_loss}

        if experiment_board is not None:
            experiment_board.on_epoch_end(
                generation=generation,
                epoch_in_generation=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                lr=applied_lr,
                param_count=sum(p.numel() for p in model.parameters()),
            )

        if not quiet and (epoch % print_every == 0 or epoch == epochs - 1):
            param_count = sum(p.numel() for p in model.parameters())
            print(
                f"Epoch: {epoch} Accuracy: {train_acc:.3f} loss: {train_loss:.3f} "
                f"val_acc: {val_acc:.3f} val_loss: {val_loss:.3f} "
                f"lr: {applied_lr:.3f} param_count: {param_count}"
            )
        if stopper.check(model, epoch, metrics):
            print(f"Stopping at epoch {epoch}")
            break

    return model, history
