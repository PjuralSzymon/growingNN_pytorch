"""Base interface for simulation scheduling policies."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import Enum


class SchedulerMode(Enum):
    ALWAYS = 0
    PROGRESS_CHECK = 1
    NEVER = 2
    SLOPE_ESTIMATION = 3
    MEAN_STANDARD_DEVIATION_STAGNATION = 4


def least_squares_slope(values: Sequence[float]) -> float:
    if len(values) < 2:
        raise ValueError("Slope estimation requires at least two values")
    x_mean = (len(values) - 1) / 2
    y_mean = sum(values) / len(values)
    numerator = sum(
        (index - x_mean) * (value - y_mean)
        for index, value in enumerate(values)
    )
    denominator = sum((index - x_mean) ** 2 for index in range(len(values)))
    return numerator / denominator


def finite_values(history: Sequence[float] | None) -> list[float] | None:
    if history is None or len(history) < 2:
        return None
    values = [float(value) for value in history]
    return values if all(math.isfinite(value) for value in values) else None


class SimulationScheduler(ABC):
    mode: SchedulerMode

    def __init__(
        self,
        simulation_time: float = 60.0,
        simulation_epochs: int = 20,
    ) -> None:
        self.simulation_time = simulation_time
        self.simulation_epochs = simulation_epochs

    @abstractmethod
    def can_simulate(
        self,
        generation: int,
        generation_val_acc: Sequence[float],
        quiet: bool = False,
    ) -> bool:
        """Return whether architecture simulation should run."""
