"""Scheduler based on training-accuracy slope within the current generation."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from math import atan, degrees

from .base import (
    SchedulerMode,
    SimulationScheduler,
    finite_values,
    least_squares_slope,
)

logger = logging.getLogger("growingnn")


class SlopeEstimationSimulationScheduler(SimulationScheduler):
    mode = SchedulerMode.SLOPE_ESTIMATION
    uses_current_generation_values = True

    def __init__(
        self,
        simulation_time: float = 60.0,
        simulation_epochs: int = 20,
        angle_threshold: float = 1.0,
    ) -> None:
        super().__init__(simulation_time, simulation_epochs)
        if not 0 <= angle_threshold <= 90:
            raise ValueError("angle_threshold must be between 0 and 90 degrees")
        self.angle_threshold = angle_threshold

    def _should_simulate(
        self,
        generation: int,
        generation_train_acc: Sequence[float],
        quiet: bool = False,
    ) -> bool:
        values = None
        if len(generation_train_acc) >= 3:
            middle = len(generation_train_acc) // 2
            values = finite_values(
                [
                    generation_train_acc[0],
                    generation_train_acc[middle],
                    generation_train_acc[-1],
                ]
            )
        if values is None:
            logger.info(
                "Slope estimation generation=%s angle=unavailable threshold=%gdeg points=start,middle,end simulation=skip",
                generation,
                self.angle_threshold,
            )
            return False
        slope = least_squares_slope(values)
        angle = degrees(atan(slope))
        can_simulate = abs(angle) <= self.angle_threshold
        logger.info(
            "Slope estimation generation=%s angle=%.3fdeg threshold=%gdeg slope=%g values=%s simulation=%s",
            generation,
            angle,
            self.angle_threshold,
            slope,
            values,
            "run" if can_simulate else "skip",
        )
        return can_simulate
