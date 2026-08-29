"""GrowingNN action-aware learning-rate schedules.

These schedules control how LR reacts to architecture actions via
``structure_changed()`` (warmup recovery) or generation-local progressive
curves. They are not the same as standard global epoch LR schedules; see
``lr_scheduler_global`` for cosine / step / exponential-style curves.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from math import cos, exp, pi, tanh

MIN_LEARNING_RATE = 0.001


def clamp_to_minimum_learning_rate(learning_rate: float) -> float:
    """Enforce the global LR floor used by every schedule path."""
    return max(MIN_LEARNING_RATE, float(learning_rate))


def compute_schedule_value_without_advancing(
    schedule: LearningRateSchedule,
    generation_local_epoch: int,
    generation_epoch_count: int,
) -> float:
    """
    Evaluate a schedule at its current logical step without mutating counters.

    Warmup schedules read iterations_since_change; generation-local schedules
    use generation_local_epoch / generation_epoch_count.
    """
    if hasattr(schedule, "iterations_since_change"):
        return float(
            schedule.compute(
                float(schedule.iterations_since_change),
                float(schedule.warmup_iterations),
            )
        )
    return float(
        schedule.compute(float(generation_local_epoch), float(generation_epoch_count))
    )


def mark_warmup_schedule_as_fully_complete(schedule: LearningRateSchedule) -> None:
    """Set action-aware warmup so the next factor is already at its peak (idle)."""
    if hasattr(schedule, "iterations_since_change") and hasattr(schedule, "warmup_iterations"):
        schedule.iterations_since_change = int(schedule.warmup_iterations)


class ScheduleMode(Enum):
    CONSTANT = 0
    PROGRESSIVE = 1
    PROGRESSIVE_PARABOLIC = 2
    WARMUP_COSINE = 3
    WARMUP_LOGISTIC = 4
    WARMUP_EXPONENTIAL = 5


class LearningRateSchedule(ABC):
    def __init__(
        self,
        alpha: float,
        steepness: float = 0.2,
        warmup_iterations: int = 100,
        k: float = 10.0,
    ):
        if alpha < 0:
            raise ValueError("Alpha must be non-negative")
        if warmup_iterations <= 0:
            raise ValueError("Warmup iterations must be positive")
        if k <= 0:
            raise ValueError("Warmup steepness k must be positive")
        self.alpha = alpha
        self.steepness = steepness
        self.warmup_iterations = warmup_iterations
        self.k = k

    def alpha_scheduler(self, i: int, iterations: int) -> float:
        return clamp_to_minimum_learning_rate(self.compute(float(i), float(iterations)))

    def unclamped_alpha_scheduler(self, i: int, iterations: int) -> float:
        """Advance and return the raw schedule value. Do not apply MIN_LEARNING_RATE."""
        return self.compute(float(i), float(iterations))

    def structure_changed(self) -> None:
        # No-op for generation-local schedules; WarmupSchedule overrides to reset.
        pass

    def reset(self) -> None:
        self.structure_changed()

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


class WarmupSchedule(LearningRateSchedule, ABC):
    def __init__(
        self,
        alpha: float,
        steepness: float = 0.2,
        warmup_iterations: int = 100,
        k: float = 10.0,
    ):
        super().__init__(alpha, steepness, warmup_iterations, k)
        self.iterations_since_change = 0

    def alpha_scheduler(self, i: int, iterations: int) -> float:
        # i / iterations are generation-local; warmup tracks epochs since last action.
        return clamp_to_minimum_learning_rate(self.unclamped_alpha_scheduler(i, iterations))

    def unclamped_alpha_scheduler(self, i: int, iterations: int) -> float:
        """Advance warmup and return the raw 0..alpha factor. Do not apply MIN_LEARNING_RATE."""
        value = self.compute(float(self.iterations_since_change), float(self.warmup_iterations))
        self.iterations_since_change += 1
        return value

    def structure_changed(self) -> None:
        self.iterations_since_change = 0

    def progress(self, i: float) -> float:
        return max(0.0, min(i / self.warmup_iterations, 1.0))


class CosineWarmupSchedule(WarmupSchedule):
    def compute(self, i: float, iterations: float) -> float:
        x = self.progress(i)
        if x >= 1.0:
            return self.alpha
        return self.alpha * (1 - cos(pi * x)) / 2


class LogisticWarmupSchedule(WarmupSchedule):
    def compute(self, i: float, iterations: float) -> float:
        x = self.progress(i)
        if x >= 1.0:
            return self.alpha
        low = (1 + tanh(-self.k / 4)) / 2
        high = (1 + tanh(self.k / 4)) / 2
        value = (1 + tanh(self.k * (x - 0.5) / 2)) / 2
        if high == low:
            return self.alpha * x
        return self.alpha * (value - low) / (high - low)


class ExponentialWarmupSchedule(WarmupSchedule):
    def compute(self, i: float, iterations: float) -> float:
        x = self.progress(i)
        if x >= 1.0:
            return self.alpha
        denominator = 1 - exp(-self.k)
        if denominator == 0:
            return self.alpha * x
        return self.alpha * (1 - exp(-self.k * x)) / denominator


_SCHEDULES: dict[ScheduleMode, type[LearningRateSchedule]] = {
    ScheduleMode.CONSTANT: ConstantSchedule,
    ScheduleMode.PROGRESSIVE: ProgressiveSchedule,
    ScheduleMode.PROGRESSIVE_PARABOLIC: ProgressiveParabolicSchedule,
    ScheduleMode.WARMUP_COSINE: CosineWarmupSchedule,
    ScheduleMode.WARMUP_LOGISTIC: LogisticWarmupSchedule,
    ScheduleMode.WARMUP_EXPONENTIAL: ExponentialWarmupSchedule,
}


class LearningRateScheduler(ABC):
    """
    Public LR scheduler interface used by training and simulation.

    Concrete kinds:
    - ``ActionLearningRateScheduler`` — GrowingNN action / generation schedules
    - ``ComposedLearningRateScheduler`` — global epoch curve interpolated with action recovery
    """

    @abstractmethod
    def alpha_scheduler(self, i: int, iterations: int) -> float:
        """Return LR for this epoch and advance schedule state."""

    @abstractmethod
    def structure_changed(self) -> None:
        """Notify the scheduler that an architecture action ran."""

    def reset(self) -> None:
        self.structure_changed()

    @abstractmethod
    def learning_rate_config_board_labels(self) -> tuple[str, float]:
        """Mode name and representative LR for ExperimentBoard snapshots."""


class ActionLearningRateScheduler(LearningRateScheduler):
    """GrowingNN action / generation-local LR schedules."""

    def __init__(
        self,
        mode: ScheduleMode,
        alpha: float,
        steepness: float = 0.2,
        warmup_iterations: int = 100,
        k: float = 10.0,
    ):
        self._schedule = _SCHEDULES[mode](alpha, steepness, warmup_iterations, k)

    def alpha_scheduler(self, i: int, iterations: int) -> float:
        return self._schedule.alpha_scheduler(i, iterations)

    def unclamped_alpha_scheduler(self, i: int, iterations: int) -> float:
        """Advance and return the raw schedule value for use as a 0..1 recovery factor."""
        return self._schedule.unclamped_alpha_scheduler(i, iterations)

    def structure_changed(self) -> None:
        self._schedule.structure_changed()

    def learning_rate_config_board_labels(self) -> tuple[str, float]:
        """Mode name and alpha for ExperimentBoard config snapshots."""
        return type(self._schedule).__name__, float(self._schedule.alpha)
