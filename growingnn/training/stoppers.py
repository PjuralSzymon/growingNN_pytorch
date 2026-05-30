"""Early stopping utilities for PyTorch training."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

import torch.fx as fx
import torch.nn as nn


class StopperMode(Enum):
    EMPTY = 0
    ACCURACY = 1
    PARAMETER_COUNT = 2
    ACCURACY_AND_REDUCTION = 3


def _parameter_count(model: nn.Module | fx.GraphModule) -> int:
    return sum(p.numel() for p in model.parameters())


class BaseStopper(ABC):
    @abstractmethod
    def check(
        self,
        model: nn.Module | fx.GraphModule,
        epoch: int,
        metrics: dict[str, float] | None = None,
    ) -> bool:
        pass

    def reset(self) -> None:
        pass


class EmptyStopper(BaseStopper):
    def check(
        self,
        model: nn.Module | fx.GraphModule,
        epoch: int,
        metrics: dict[str, float] | None = None,
    ) -> bool:
        return False


class AccuracyStopper(BaseStopper):
    def __init__(self, target_accuracy: float = 0.9, metric_name: str = "accuracy"):
        self.target_accuracy = target_accuracy
        self.metric_name = metric_name

    def check(
        self,
        model: nn.Module | fx.GraphModule,
        epoch: int,
        metrics: dict[str, float] | None = None,
    ) -> bool:
        if metrics is None or "accuracy" not in metrics:
            return False
        current_accuracy = metrics["accuracy"]
        if current_accuracy >= self.target_accuracy:
            msg = (
                f"Stopping: {self.metric_name} reached {current_accuracy:.4f} "
                f"(target: {self.target_accuracy:.4f}) at epoch {epoch}"
            )
            print(msg)
            return True
        return False


class ParameterCountStopper(BaseStopper):
    def __init__(
        self,
        decrease_threshold: float = 0.5,
        metric_name: str = "parameter_count",
    ):
        self.decrease_threshold = decrease_threshold
        self.metric_name = metric_name
        self.initial_parameter_count: int | None = None

    def check(
        self,
        model: nn.Module | fx.GraphModule,
        epoch: int,
        metrics: dict[str, float] | None = None,
    ) -> bool:
        current_param_count = _parameter_count(model)
        if self.initial_parameter_count is None:
            self.initial_parameter_count = current_param_count
            return False
        if self.initial_parameter_count <= 0:
            return False

        decrease_ratio = (
            self.initial_parameter_count - current_param_count
        ) / self.initial_parameter_count
        if decrease_ratio >= self.decrease_threshold:
            msg = (
                f"Stopping: {self.metric_name} decreased by {decrease_ratio:.2%} from initial "
                f"(from {self.initial_parameter_count} to {current_param_count}) at epoch {epoch}"
            )
            print(msg)
            return True
        return False

    def reset(self) -> None:
        self.initial_parameter_count = None


class AccuracyAndReductionStopper(BaseStopper):
    def __init__(
        self,
        target_accuracy: float = 0.9,
        parameter_decrease_threshold: float = 0.5,
    ):
        self.target_accuracy = target_accuracy
        self.parameter_decrease_threshold = parameter_decrease_threshold
        self.parameter_stopper = ParameterCountStopper(
            decrease_threshold=parameter_decrease_threshold
        )

    def check(
        self,
        model: nn.Module | fx.GraphModule,
        epoch: int,
        metrics: dict[str, float] | None = None,
    ) -> bool:
        accuracy_reached = (
            metrics is not None
            and metrics.get("accuracy", 0.0) >= self.target_accuracy
        )
        current_param_count = _parameter_count(model)
        if self.parameter_stopper.initial_parameter_count is None:
            self.parameter_stopper.initial_parameter_count = current_param_count
            return False

        initial = self.parameter_stopper.initial_parameter_count
        decrease_ratio = (
            (initial - current_param_count) / initial if initial > 0 else 0.0
        )
        parameter_reduced = decrease_ratio >= self.parameter_decrease_threshold
        if accuracy_reached and parameter_reduced:
            msg = (
                f"Stopping: accuracy >= {self.target_accuracy:.2f} "
                f"and parameters reduced by >= {self.parameter_decrease_threshold:.1%} "
                f"at epoch {epoch}"
            )
            print(msg)
            return True
        return False

    def reset(self) -> None:
        self.parameter_stopper.reset()


def _build_stopper(
    mode: StopperMode,
    target_accuracy: float,
    parameter_decrease_threshold: float,
) -> BaseStopper:
    if mode == StopperMode.EMPTY:
        return EmptyStopper()
    if mode == StopperMode.ACCURACY:
        return AccuracyStopper(target_accuracy=target_accuracy)
    if mode == StopperMode.PARAMETER_COUNT:
        return ParameterCountStopper(decrease_threshold=parameter_decrease_threshold)
    return AccuracyAndReductionStopper(
        target_accuracy=target_accuracy,
        parameter_decrease_threshold=parameter_decrease_threshold,
    )


class TrainingStopper:
    def __init__(
        self,
        mode: StopperMode,
        target_accuracy: float = 0.9,
        parameter_decrease_threshold: float = 0.5,
    ):
        self._stopper = _build_stopper(
            mode, target_accuracy, parameter_decrease_threshold
        )

    def check(
        self,
        model: nn.Module | fx.GraphModule,
        epoch: int,
        metrics: dict[str, float] | None = None,
    ) -> bool:
        return self._stopper.check(model, epoch, metrics)

    def reset(self) -> None:
        self._stopper.reset()
