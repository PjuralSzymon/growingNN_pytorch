"""Unit tests for ``growingnn.training.lr_scheduler``."""

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from growingnn.training.lr_scheduler import (
    ConstantSchedule,
    LearningRateScheduler,
    ProgressiveParabolicSchedule,
    ProgressiveSchedule,
    ScheduleMode,
)


def test_constant_schedule_returns_alpha_for_any_iteration():
    """
    ConstantSchedule should return the base learning rate unchanged.
    """
    # Arrange
    schedule = ConstantSchedule(alpha=0.02)
    iterations = 100.0

    # Act
    result_early = schedule.alpha_scheduler(int(iterations * 0.1), int(iterations))
    result_mid = schedule.alpha_scheduler(int(iterations * 0.5), int(iterations))
    result_late = schedule.alpha_scheduler(int(iterations * 0.9), int(iterations))

    # Assert
    assert result_early == result_mid == result_late == 0.02


def test_progressive_schedule_ramps_up_before_threshold():
    """
    ProgressiveSchedule should scale alpha by (i+1)/(thresh+2) before the ramp threshold.
    """
    # Arrange
    schedule = ProgressiveSchedule(alpha=1.0, steepness=0.2)
    iterations = 100.0

    # Act
    result_init = schedule.compute(int(iterations * 0.1), iterations)
    result_mid = schedule.compute(int(iterations * 0.5), iterations)
    result_pick = schedule.compute(int(iterations * 0.2), iterations)
    result_end = schedule.compute(int(iterations * 0.9), iterations)

    # Assert
    assert result_init < result_mid
    assert result_mid > result_end
    assert result_pick > result_mid


def test_progressive_schedule_decays_after_threshold():
    """
    ProgressiveSchedule should linearly decay alpha after the ramp threshold.
    """
    # Arrange
    schedule = ProgressiveSchedule(alpha=1.0, steepness=0.2)
    iterations = 100.0

    # Act
    result_early_decay = schedule.compute(int(iterations * 0.3), iterations)
    result_mid_decay = schedule.compute(int(iterations * 0.5), iterations)
    result_late_decay = schedule.compute(int(iterations * 0.9), iterations)

    # Assert
    assert result_early_decay > result_mid_decay
    assert result_mid_decay > result_late_decay


def test_progressive_parabolic_schedule_is_zero_at_start():
    """
    ProgressiveParabolicSchedule should return zero at iteration zero.
    """
    # Arrange
    schedule = ProgressiveParabolicSchedule(alpha=1.0, steepness=0.2)
    iterations = 100.0

    # Act
    result_start = schedule.compute(0, iterations)
    result_early = schedule.compute(int(iterations * 0.05), iterations)
    result_ramp = schedule.compute(int(iterations * 0.1), iterations)

    # Assert
    assert result_start == 0.0
    assert result_start < result_early
    assert result_early < result_ramp


def test_progressive_parabolic_schedule_peaks_before_threshold():
    """
    ProgressiveParabolicSchedule should follow the parabolic ramp before the threshold.
    """
    # Arrange
    schedule = ProgressiveParabolicSchedule(alpha=1.0, steepness=0.2)
    iterations = 100.0

    # Act
    result_early = schedule.compute(int(iterations * 0.05), iterations)
    result_mid = schedule.compute(int(iterations * 0.1), iterations)
    result_late_ramp = schedule.compute(int(iterations * 0.15), iterations)

    # Assert
    assert result_early < result_mid
    assert result_mid < result_late_ramp


def test_learning_rate_scheduler_rejects_negative_alpha():
    """
    LearningRateScheduler should reject a negative base learning rate.
    """
    # Arrange
    mode = ScheduleMode.CONSTANT
    alpha = -0.5

    # Act / Assert
    with pytest.raises(ValueError, match="Alpha must be non-negative"):
        LearningRateScheduler(mode, alpha=alpha)


def test_learning_rate_scheduler_picks_schedule_from_mode():
    """
    LearningRateScheduler should dispatch to the schedule class matching ScheduleMode.
    """
    # Arrange
    constant = LearningRateScheduler(ScheduleMode.CONSTANT, alpha=0.03)
    progressive = LearningRateScheduler(ScheduleMode.PROGRESSIVE, alpha=0.03, steepness=0.2)
    iterations = 100.0

    # Act
    result_constant_early = constant.alpha_scheduler(int(iterations * 0.1), int(iterations))
    result_constant_late = constant.alpha_scheduler(int(iterations * 0.9), int(iterations))
    result_progressive = progressive.alpha_scheduler(int(iterations * 0.5), int(iterations))

    # Assert
    assert result_constant_early == result_constant_late == 0.03
    assert result_progressive != result_constant_early


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
