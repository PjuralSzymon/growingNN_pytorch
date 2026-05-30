from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum


class ScheduleMode(Enum):
    CONSTANT = 0
    PROGRESSIVE = 1
    PROGRESSIVE_PARABOLIC = 2


class LearningRateSchedule(ABC):
    def __init__(self, alpha: float, steepness: float = 0.2):
        self.alpha = alpha
        self.steepness = steepness

    def alpha_scheduler(self, i: int, iterations: int) -> float:
        lr = self.compute(float(i), float(iterations))
        return max(0.0, lr)

    @abstractmethod
    def compute(self, i: float, iterations: float) -> float:
        pass


class ConstantSchedule(LearningRateSchedule):
    def compute(self, i: float, iterations: float) -> float:
        return self.alpha


class ProgressiveSchedule(LearningRateSchedule):
    def compute(self, i: float, iterations: float) -> float:
        thresh = self.steepness * iterations
        if i < thresh:
            return self.alpha * ((i + 1) / (thresh + 2))
        return self.alpha * (1 - (i - thresh) / (iterations - thresh + 2))


class ProgressiveParabolicSchedule(LearningRateSchedule):
    def compute(self, i: float, iterations: float) -> float:
        thresh = self.steepness * iterations
        if i < thresh:
            return self.alpha * (-(1 / thresh**2) * (i - thresh) ** 2 + 1)
        return self.alpha * (-(1 / (iterations - thresh) ** 2) * (i - thresh) ** 2 + 1)


_SCHEDULES: dict[ScheduleMode, type[LearningRateSchedule]] = {
    ScheduleMode.CONSTANT: ConstantSchedule,
    ScheduleMode.PROGRESSIVE: ProgressiveSchedule,
    ScheduleMode.PROGRESSIVE_PARABOLIC: ProgressiveParabolicSchedule,
}


class LearningRateScheduler:
    def __init__(self, mode: ScheduleMode, alpha: float, steepness: float = 0.2):
        self._schedule = _SCHEDULES[mode](alpha, steepness)

    def alpha_scheduler(self, i: int, iterations: int) -> float:
        return self._schedule.alpha_scheduler(i, iterations)
