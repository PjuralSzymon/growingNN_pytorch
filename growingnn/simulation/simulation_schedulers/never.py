"""Scheduler that disables simulation."""

from __future__ import annotations

from collections.abc import Sequence

from .base import SchedulerMode, SimulationScheduler


class NeverSimulationScheduler(SimulationScheduler):
    mode = SchedulerMode.NEVER

    def _should_simulate(
        self,
        generation: int,
        generation_val_acc: Sequence[float],
        quiet: bool = False,
    ) -> bool:
        return False
