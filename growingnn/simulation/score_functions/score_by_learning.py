"""Learning-based simulation score terms."""

from __future__ import annotations

from enum import Enum

import torch.fx as fx
import torch.nn as nn

from growingnn.core.config import RunningConfig
from growingnn.training.gradient_descent import gradient_descent


class AccuracyMetric(str, Enum):
    """Which learning metric simulation uses when grading a candidate."""

    VAL_ACC = "val_acc"
    TRAIN_ACC = "train_acc"


def parse_accuracy_metric(value: AccuracyMetric | str | None) -> AccuracyMetric:
    """Normalize config/HP values to AccuracyMetric. Default remains validation."""
    if value is None:
        return AccuracyMetric.VAL_ACC
    if isinstance(value, AccuracyMetric):
        return value
    normalized = str(value).strip().lower()
    aliases = {
        "val": AccuracyMetric.VAL_ACC,
        "val_acc": AccuracyMetric.VAL_ACC,
        "validation": AccuracyMetric.VAL_ACC,
        "validation_acc": AccuracyMetric.VAL_ACC,
        "train": AccuracyMetric.TRAIN_ACC,
        "train_acc": AccuracyMetric.TRAIN_ACC,
        "training": AccuracyMetric.TRAIN_ACC,
        "training_acc": AccuracyMetric.TRAIN_ACC,
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unknown accuracy metric {value!r}. Expected one of: "
            f"{', '.join(sorted(aliases))}"
        )
    return aliases[normalized]


def _accuracy_metric(running_config: RunningConfig) -> AccuracyMetric:
    score = running_config.simulation_score
    raw = getattr(score, "accuracy_metric", AccuracyMetric.VAL_ACC)
    return parse_accuracy_metric(raw)


def _history_keys(metric: AccuracyMetric) -> tuple[str, str]:
    if metric is AccuracyMetric.TRAIN_ACC:
        return "train_acc", "train_loss"
    return "val_acc", "val_loss"


def score_acc(
    model: nn.Module | fx.GraphModule,
    running_config: RunningConfig,
) -> float:
    metric = _accuracy_metric(running_config)
    acc_key, _ = _history_keys(metric)
    _, history = gradient_descent(
        model,
        running_config.simulation_scheduler.simulation_epochs,
        running_config.sim_train_loader,
        running_config.sim_val_loader,
        running_config.criterion,
        running_config.lr_scheduler,
        quiet=True,
        device=running_config.device,
    )
    return float(history[acc_key][-1])


def score_loss(
    model: nn.Module | fx.GraphModule,
    running_config: RunningConfig,
) -> float:
    metric = _accuracy_metric(running_config)
    _, loss_key = _history_keys(metric)
    _, history = gradient_descent(
        model,
        running_config.simulation_scheduler.simulation_epochs,
        running_config.sim_train_loader,
        running_config.sim_val_loader,
        running_config.criterion,
        running_config.lr_scheduler,
        quiet=True,
        device=running_config.device,
    )
    loss = float(history[loss_key][-1])
    return min(1.0 / (max(loss, 1e-8) + 1), 1.0)
