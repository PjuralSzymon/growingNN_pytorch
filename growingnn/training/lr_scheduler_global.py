"""Standard global learning-rate schedules (epoch-indexed curves).

These are pure ``lr_at(global_epoch, total_epochs)`` adapters in the spirit of
``torch.optim.lr_scheduler`` (cosine, step, exponential, linear, constant).
They do not react to GrowingNN architecture actions.

``ComposedLearningRateScheduler`` multiplies a global schedule by the
action-aware recovery factor from ``lr_scheduler_action``.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from math import cos, pi
from typing import Any, Iterator, Protocol, runtime_checkable

from growingnn.training.lr_scheduler_action import (
    ActionLearningRateScheduler,
    LearningRateScheduler,
    ScheduleMode,
    clamp_to_minimum_learning_rate,
    mark_warmup_schedule_as_fully_complete,
)


def freeze_global_learning_rate_progress_if_supported(lr_scheduler: Any):
    """
    Context manager that freezes global epoch progress when available.

    Used by simulation scoring so shared composed schedulers do not burn
    global_epoch during short rollouts. Non-composed schedulers are a no-op.
    """
    freeze = getattr(lr_scheduler, "freeze_global_schedule_progress", None)
    if callable(freeze):
        return freeze()
    return nullcontext()


@runtime_checkable
class GlobalLearningRateSchedule(Protocol):
    """Global training LR curve as a pure function of epoch index."""

    def lr_at(self, global_epoch: int, total_epochs: int) -> float:
        """Return absolute learning rate at *global_epoch*."""


class ConstantLearningRate:
    """Constant learning rate for the whole run."""

    def __init__(self, lr: float):
        if lr < 0:
            raise ValueError("lr must be non-negative")
        self.lr = lr

    def lr_at(self, global_epoch: int, total_epochs: int) -> float:
        return self.lr


class CosineAnnealingLearningRate:
    """Cosine decay from *initial_lr* to *eta_min* over *T_max* epochs."""

    def __init__(self, T_max: int, eta_min: float = 0.0, initial_lr: float = 0.01):
        if T_max <= 0:
            raise ValueError("T_max must be positive")
        if eta_min < 0 or initial_lr < 0:
            raise ValueError("learning rates must be non-negative")
        self.T_max = T_max
        self.eta_min = eta_min
        self.initial_lr = initial_lr

    def lr_at(self, global_epoch: int, total_epochs: int) -> float:
        epoch = max(0, min(int(global_epoch), self.T_max))
        # Match torch.optim.lr_scheduler.CosineAnnealingLR for last_epoch == epoch.
        return self.eta_min + (self.initial_lr - self.eta_min) * (
            1 + cos(pi * epoch / self.T_max)
        ) / 2


class StepLearningRate:
    """Multiply *initial_lr* by *gamma* every *step_size* epochs."""

    def __init__(self, step_size: int, gamma: float = 0.1, initial_lr: float = 0.01):
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        if gamma < 0 or initial_lr < 0:
            raise ValueError("gamma and initial_lr must be non-negative")
        self.step_size = step_size
        self.gamma = gamma
        self.initial_lr = initial_lr

    def lr_at(self, global_epoch: int, total_epochs: int) -> float:
        epoch = max(0, int(global_epoch))
        return self.initial_lr * (self.gamma ** (epoch // self.step_size))


class ExponentialLearningRate:
    """Multiply *initial_lr* by *gamma* each epoch: lr = initial_lr * gamma^epoch."""

    def __init__(self, gamma: float, initial_lr: float = 0.01):
        if gamma < 0 or initial_lr < 0:
            raise ValueError("gamma and initial_lr must be non-negative")
        self.gamma = gamma
        self.initial_lr = initial_lr

    def lr_at(self, global_epoch: int, total_epochs: int) -> float:
        epoch = max(0, int(global_epoch))
        return self.initial_lr * (self.gamma ** epoch)


class LinearDecayLearningRate:
    """Linear decay from *initial_lr* to *eta_min* over *T_max* epochs."""

    def __init__(self, T_max: int, eta_min: float = 0.0, initial_lr: float = 0.01):
        if T_max <= 0:
            raise ValueError("T_max must be positive")
        if eta_min < 0 or initial_lr < 0:
            raise ValueError("learning rates must be non-negative")
        self.T_max = T_max
        self.eta_min = eta_min
        self.initial_lr = initial_lr

    def lr_at(self, global_epoch: int, total_epochs: int) -> float:
        epoch = max(0, min(int(global_epoch), self.T_max))
        progress = epoch / self.T_max
        return self.initial_lr + (self.eta_min - self.initial_lr) * progress


def build_global_learning_rate_schedule(
    schedule_name: str,
    *,
    total_epochs: int,
    initial_lr: float,
    eta_min: float = 1e-4,
    step_size: int | None = None,
    gamma: float = 0.1,
) -> GlobalLearningRateSchedule:
    """
    Build one absolute epoch-based schedule by name.

    schedule_name: "cosine" | "step" | "exponential" | "constant" | "linear"
    """
    name = str(schedule_name).strip().lower()
    if name in ("cosine", "cosineannealing", "cosine_annealing"):
        return CosineAnnealingLearningRate(
            T_max=total_epochs, eta_min=eta_min, initial_lr=initial_lr
        )
    if name in ("step", "steplr"):
        return StepLearningRate(
            step_size=step_size or max(1, total_epochs // 3),
            gamma=gamma,
            initial_lr=initial_lr,
        )
    if name in ("exponential", "explr", "exponentiallr"):
        return ExponentialLearningRate(gamma=gamma, initial_lr=initial_lr)
    if name in ("linear", "linear_decay", "lineardecay"):
        return LinearDecayLearningRate(
            T_max=total_epochs, eta_min=eta_min, initial_lr=initial_lr
        )
    if name == "constant":
        return ConstantLearningRate(lr=initial_lr)
    raise ValueError(
        f"Unknown global schedule {schedule_name!r}. "
        "Expected cosine, step, exponential, linear, or constant."
    )


def build_composed_learning_rate_scheduler(
    schedule_name: str,
    *,
    total_epochs: int,
    initial_lr: float,
    warmup_iterations: int = 10,
    k: float = 10.0,
    eta_min: float = 1e-4,
    step_size: int | None = None,
    gamma: float = 0.1,
) -> ComposedLearningRateScheduler:
    """
    Build global×action-recovery composition for training or experiment scripts.

    Recovery is WARMUP_LOGISTIC with alpha=1.0 (peak factor, not absolute LR).
    """
    return ComposedLearningRateScheduler(
        global_schedule=build_global_learning_rate_schedule(
            schedule_name,
            total_epochs=total_epochs,
            initial_lr=initial_lr,
            eta_min=eta_min,
            step_size=step_size,
            gamma=gamma,
        ),
        recovery=ActionLearningRateScheduler(
            ScheduleMode.WARMUP_LOGISTIC,
            alpha=1.0,
            warmup_iterations=warmup_iterations,
            k=k,
        ),
        total_epochs=total_epochs,
        initial_lr=initial_lr,
    )


class ComposedLearningRateScheduler(LearningRateScheduler):
    """
    Multiply a global LR curve by GrowingNN action recovery.

    effective_lr = max(MIN_LEARNING_RATE, global_lr(global_epoch) * recovery_factor)

    Recovery must use alpha=1.0 so its output is a 0..1 factor. Until the first
    structure_changed(), recovery stays fully warmed (factor ≈ 1) so training
    follows the global schedule only.
    """

    def __init__(
        self,
        global_schedule: GlobalLearningRateSchedule,
        recovery: ActionLearningRateScheduler,
        *,
        total_epochs: int,
        initial_lr: float | None = None,
    ):
        if total_epochs <= 0:
            raise ValueError("total_epochs must be positive")
        recovery_alpha = float(recovery._schedule.alpha)
        if recovery_alpha != 1.0:
            raise ValueError(
                "ComposedLearningRateScheduler recovery must use alpha=1.0 "
                f"(peak multiplier); got alpha={recovery_alpha}"
            )
        self.global_schedule = global_schedule
        self.recovery = recovery
        self.total_epochs = total_epochs
        self.initial_lr = (
            float(initial_lr)
            if initial_lr is not None
            else float(global_schedule.lr_at(0, total_epochs))
        )
        self.global_epoch = 0
        self._global_schedule_progress_frozen = False
        mark_warmup_schedule_as_fully_complete(self.recovery._schedule)

    def _compose_effective_learning_rate(
        self,
        global_learning_rate: float,
        recovery_factor: float,
    ) -> float:
        return clamp_to_minimum_learning_rate(global_learning_rate * recovery_factor)

    def alpha_scheduler(self, i: int, iterations: int) -> float:
        global_learning_rate = self.global_schedule.lr_at(
            self.global_epoch, self.total_epochs
        )
        recovery_factor = self.recovery.alpha_scheduler(i, iterations)
        if not self._global_schedule_progress_frozen:
            self.global_epoch += 1
        return self._compose_effective_learning_rate(global_learning_rate, recovery_factor)

    def structure_changed(self) -> None:
        self.recovery.structure_changed()

    def reset(self) -> None:
        self.structure_changed()

    def learning_rate_config_board_labels(self) -> tuple[str, float]:
        """Mode name and initial global LR for ExperimentBoard config snapshots."""
        recovery_schedule = self.recovery._schedule
        mode_name = (
            f"Composed[{type(self.global_schedule).__name__}"
            f"+{type(recovery_schedule).__name__}]"
        )
        return mode_name, float(self.initial_lr)

    @contextmanager
    def freeze_global_schedule_progress(self) -> Iterator[ComposedLearningRateScheduler]:
        """Freeze global epoch progress (use during simulation scoring)."""
        previous = self._global_schedule_progress_frozen
        self._global_schedule_progress_frozen = True
        try:
            yield self
        finally:
            self._global_schedule_progress_frozen = previous
