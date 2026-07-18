"""Scheduler based on progress between generations."""

from __future__ import annotations

from collections.abc import Sequence

from .base import SchedulerMode, SimulationScheduler


class ProgressCheckSimulationScheduler(SimulationScheduler):
    mode = SchedulerMode.PROGRESS_CHECK

    def __init__(
        self,
        simulation_time: float = 60.0,
        simulation_epochs: int = 20,
        stagnation_window: int = 1,
    ) -> None:
        super().__init__(simulation_time, simulation_epochs)
        if stagnation_window < 1:
            raise ValueError("stagnation_window must be at least 1")
        self.stagnation_window = stagnation_window

    def can_simulate(
        self,
        generation: int,
        generation_val_acc: Sequence[float],
        quiet: bool = False,
    ) -> bool:
        if len(generation_val_acc) < self.stagnation_window + 1:
            return False
        recent = generation_val_acc[-(self.stagnation_window + 1):]
        return recent[-1] <= max(recent[:-1])
