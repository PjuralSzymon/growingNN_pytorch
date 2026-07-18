"""Scheduler based on final validation accuracy across generations."""

from __future__ import annotations

from collections.abc import Sequence

from .base import (
    SchedulerMode,
    SimulationScheduler,
    finite_values,
    least_squares_slope,
)


class SlopeEstimationSimulationScheduler(SimulationScheduler):
    mode = SchedulerMode.SLOPE_ESTIMATION

    def __init__(
        self,
        simulation_time: float = 60.0,
        simulation_epochs: int = 20,
        slope_epsilon: float = 1e-4,
    ) -> None:
        super().__init__(simulation_time, simulation_epochs)
        if slope_epsilon < 0:
            raise ValueError("slope_epsilon must be non-negative")
        self.slope_epsilon = slope_epsilon

    def can_simulate(
        self,
        generation: int,
        generation_val_acc: Sequence[float],
        quiet: bool = False,
    ) -> bool:
        values = finite_values(generation_val_acc)
        if values is None:
            return False
        return abs(least_squares_slope(values)) <= self.slope_epsilon
