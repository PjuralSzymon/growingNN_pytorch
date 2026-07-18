"""Detect stagnation from the mean and standard deviation of validation accuracy."""

from __future__ import annotations

import statistics
from collections.abc import Sequence

from .base import (
    SchedulerMode,
    SimulationScheduler,
    finite_values,
    least_squares_slope,
)


class MeanStandardDeviationStagnationSimulationScheduler(SimulationScheduler):
    mode = SchedulerMode.MEAN_STANDARD_DEVIATION_STAGNATION

    def __init__(
        self,
        simulation_time: float = 60.0,
        simulation_epochs: int = 20,
        slope_epsilon: float = 1e-4,
        standard_deviation_multiplier: float = 1.5,
    ) -> None:
        super().__init__(simulation_time, simulation_epochs)
        if slope_epsilon < 0:
            raise ValueError("slope_epsilon must be non-negative")
        if standard_deviation_multiplier <= 0:
            raise ValueError("standard_deviation_multiplier must be positive")
        self.slope_epsilon = slope_epsilon
        self.standard_deviation_multiplier = standard_deviation_multiplier

    def can_simulate(
        self,
        generation: int,
        generation_val_acc: Sequence[float],
        quiet: bool = False,
    ) -> bool:
        values = finite_values(generation_val_acc)
        if values is None:
            return False
        deviation = statistics.stdev(values)
        if deviation == 0:
            return True
        distance = max(values) - statistics.mean(values)
        slope = least_squares_slope(values)
        return (
            distance <= self.standard_deviation_multiplier * deviation
            and abs(slope) <= self.slope_epsilon
        )
