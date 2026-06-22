"""Decide when architecture search should run between training generations."""

from __future__ import annotations

from enum import Enum

from growingnn.core.logger import logger


class SchedulerMode(Enum):
    ALWAYS = 0
    PROGRESS_CHECK = 1
    NEVER = 2


class SimulationScheduler:
    def __init__(
        self,
        mode: SchedulerMode = SchedulerMode.PROGRESS_CHECK,
        simulation_time: float = 60.0,
        simulation_epochs: int = 20,
        stagnation_window: int = 1,
    ):
        self.mode = mode
        self.simulation_time = simulation_time
        self.simulation_epochs = simulation_epochs
        self.stagnation_window = stagnation_window

    def can_simulate(
        self,
        generation: int,
        generation_val_acc: list[float],
        *,
        quiet: bool = False,
    ) -> bool:
        if self.mode == SchedulerMode.NEVER:
            return False
        if self.mode == SchedulerMode.ALWAYS:
            return True
        if len(generation_val_acc) < self.stagnation_window + 1:
            return False
        recent = generation_val_acc[-(self.stagnation_window + 1):]
        improved = recent[-1] > max(recent[:-1])
        if not quiet and not improved:
            logger.info("Validation accuracy did not improve; running architecture simulation")
        return not improved
