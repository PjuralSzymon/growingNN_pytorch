"""Unit tests for action-aware and global learning-rate schedules."""

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from math import cos, pi

from growingnn.training.lr_scheduler_action import (
    MIN_LEARNING_RATE,
    ActionLearningRateScheduler,
    ConstantSchedule,
    LearningRateScheduler,
    ProgressiveParabolicSchedule,
    ProgressiveSchedule,
    ScheduleMode,
    clamp_to_minimum_learning_rate,
)
from growingnn.training.lr_scheduler_global import (
    ComposedLearningRateScheduler,
    ConstantLearningRate,
    CosineAnnealingLearningRate,
    ExponentialLearningRate,
    LinearDecayLearningRate,
    StepLearningRate,
    build_composed_learning_rate_scheduler,
    freeze_global_learning_rate_progress_if_supported,
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
        ActionLearningRateScheduler(mode, alpha=alpha)


def test_learning_rate_scheduler_picks_schedule_from_mode():
    """
    LearningRateScheduler should dispatch to the schedule class matching ScheduleMode.
    """
    # Arrange
    constant = ActionLearningRateScheduler(ScheduleMode.CONSTANT, alpha=0.03)
    progressive = ActionLearningRateScheduler(ScheduleMode.PROGRESSIVE, alpha=0.03, steepness=0.2)
    iterations = 100.0

    # Act
    result_constant_early = constant.alpha_scheduler(int(iterations * 0.1), int(iterations))
    result_constant_late = constant.alpha_scheduler(int(iterations * 0.9), int(iterations))
    result_progressive = progressive.alpha_scheduler(int(iterations * 0.5), int(iterations))

    # Assert
    assert result_constant_early == result_constant_late == 0.03
    assert result_progressive != result_constant_early


def test_cosine_annealing_base_matches_formula_at_endpoints():
    """
    CosineAnnealingLearningRate should equal initial_lr at epoch 0 and eta_min at t_max.
    """
    # Arrange
    base = CosineAnnealingLearningRate(t_max=100, eta_min=0.001, initial_lr=0.01)

    # Act
    lr_start = base.lr_at(0, 100)
    lr_mid = base.lr_at(50, 100)
    lr_end = base.lr_at(100, 100)
    expected_mid = 0.001 + (0.01 - 0.001) * (1 + cos(pi * 50 / 100)) / 2

    # Assert
    assert lr_start == pytest.approx(0.01)
    assert lr_end == pytest.approx(0.001)
    assert lr_mid == pytest.approx(expected_mid)


def test_step_lr_base_drops_by_gamma_each_step():
    """
    StepLearningRate should keep initial_lr until step_size, then multiply by gamma.
    """
    # Arrange
    base = StepLearningRate(step_size=10, gamma=0.1, initial_lr=0.01)

    # Act / Assert
    assert base.lr_at(0, 100) == pytest.approx(0.01)
    assert base.lr_at(9, 100) == pytest.approx(0.01)
    assert base.lr_at(10, 100) == pytest.approx(0.001)
    assert base.lr_at(20, 100) == pytest.approx(0.0001)


def test_exponential_lr_base_applies_gamma_each_epoch():
    """
    ExponentialLearningRate should return initial_lr * gamma^epoch.
    """
    # Arrange
    base = ExponentialLearningRate(gamma=0.9, initial_lr=0.01)

    # Act / Assert
    assert base.lr_at(0, 100) == pytest.approx(0.01)
    assert base.lr_at(1, 100) == pytest.approx(0.009)
    assert base.lr_at(2, 100) == pytest.approx(0.0081)


def test_composed_without_action_equals_base_curve():
    """
    With idle recovery (factor 1), composed LR should match the base schedule exactly.
    """
    # Arrange
    total = 20
    base = CosineAnnealingLearningRate(t_max=total, eta_min=0.001, initial_lr=0.01)
    composed = ComposedLearningRateScheduler(
        global_schedule=base,
        recovery=ActionLearningRateScheduler(
            ScheduleMode.WARMUP_LOGISTIC, alpha=1.0, warmup_iterations=5, k=10.0
        ),
        total_epochs=total,
        initial_lr=0.01,
    )

    # Act / Assert
    for epoch in range(total):
        expected = max(MIN_LEARNING_RATE, base.lr_at(epoch, total))
        actual = composed.alpha_scheduler(epoch % 10, 10)
        assert actual == pytest.approx(expected)


def test_composed_after_structure_changed_ramps_to_current_base():
    """
    After structure_changed, LR should start near the floor and ramp to the current base value.
    """
    # Arrange
    total = 50
    base = ConstantLearningRate(lr=0.7)
    composed = ComposedLearningRateScheduler(
        global_schedule=base,
        recovery=ActionLearningRateScheduler(
            ScheduleMode.WARMUP_LOGISTIC, alpha=1.0, warmup_iterations=10, k=10.0
        ),
        total_epochs=total,
        initial_lr=0.7,
    )
    for _ in range(15):
        composed.alpha_scheduler(0, 10)

    # Act
    composed.structure_changed()
    lr_right_after = composed.alpha_scheduler(0, 10)
    for _ in range(9):
        composed.alpha_scheduler(0, 10)
    lr_after_warmup = composed.alpha_scheduler(0, 10)

    # Assert
    assert lr_right_after < 0.2
    assert lr_after_warmup == pytest.approx(0.7)


def test_composed_rejects_recovery_alpha_not_one():
    """
    ComposedLearningRateScheduler should require recovery.alpha == 1.0.
    """
    # Arrange
    recovery = ActionLearningRateScheduler(ScheduleMode.CONSTANT, alpha=0.01)
    global_schedule = ConstantLearningRate(lr=0.01)

    # Act / Assert
    with pytest.raises(ValueError, match="alpha=1.0"):
        ComposedLearningRateScheduler(
            global_schedule=global_schedule,
            recovery=recovery,
            total_epochs=10,
        )


def test_composed_learning_rate_scheduler_is_learning_rate_scheduler():
    """
    ComposedLearningRateScheduler should be a LearningRateScheduler subtype.
    """
    # Arrange
    composed = ComposedLearningRateScheduler(
        global_schedule=ConstantLearningRate(lr=0.01),
        recovery=ActionLearningRateScheduler(ScheduleMode.CONSTANT, alpha=1.0),
        total_epochs=10,
    )

    # Act / Assert
    assert isinstance(composed, LearningRateScheduler)
    assert isinstance(composed, ComposedLearningRateScheduler)


def test_composed_freeze_global_schedule_progress_does_not_advance_global_epoch():
    """
    freeze_global_schedule_progress should keep global_epoch unchanged while alpha_scheduler still returns a value.
    """
    # Arrange
    composed = ComposedLearningRateScheduler(
        global_schedule=StepLearningRate(step_size=5, gamma=0.5, initial_lr=0.01),
        recovery=ActionLearningRateScheduler(ScheduleMode.CONSTANT, alpha=1.0),
        total_epochs=20,
        initial_lr=0.01,
    )
    composed.alpha_scheduler(0, 10)
    epoch_before = composed.global_epoch

    # Act
    with composed.freeze_global_schedule_progress():
        lr_a = composed.alpha_scheduler(0, 10)
        lr_b = composed.alpha_scheduler(0, 10)
    epoch_after = composed.global_epoch
    lr_next = composed.alpha_scheduler(0, 10)

    # Assert
    assert epoch_before == epoch_after == 1
    assert lr_a == lr_b == pytest.approx(0.01)
    assert composed.global_epoch == 2
    assert lr_next == pytest.approx(0.01)


def test_non_composed_learning_rate_scheduler_unchanged():
    """
    Standalone LearningRateScheduler CONSTANT behavior should remain absolute alpha.
    """
    # Arrange
    scheduler = ActionLearningRateScheduler(ScheduleMode.CONSTANT, alpha=0.05)

    # Act
    values = [scheduler.alpha_scheduler(i, 10) for i in range(5)]

    # Assert
    assert values == [0.05] * 5


def test_linear_decay_base_matches_endpoints():
    """
    LinearDecayLearningRate should equal initial_lr at epoch 0 and eta_min at t_max.
    """
    # Arrange
    base = LinearDecayLearningRate(t_max=100, eta_min=0.001, initial_lr=0.01)

    # Act / Assert
    assert base.lr_at(0, 100) == pytest.approx(0.01)
    assert base.lr_at(50, 100) == pytest.approx(0.0055)
    assert base.lr_at(100, 100) == pytest.approx(0.001)


def test_clamp_to_minimum_learning_rate_enforces_floor():
    """
    clamp_to_minimum_learning_rate should raise values below MIN_LEARNING_RATE up to the floor.
    """
    # Arrange / Act / Assert
    assert clamp_to_minimum_learning_rate(0.0001) == MIN_LEARNING_RATE
    assert clamp_to_minimum_learning_rate(0.05) == 0.05


def test_build_composed_learning_rate_scheduler_selects_cosine_base():
    """
    build_composed_learning_rate_scheduler should wire CosineAnnealingLearningRate with recovery alpha 1.0.
    """
    # Arrange / Act
    composed = build_composed_learning_rate_scheduler(
        "cosine", total_epochs=30, initial_lr=0.02, eta_min=0.001
    )

    # Assert
    assert isinstance(composed.global_schedule, CosineAnnealingLearningRate)
    assert composed.initial_lr == 0.02
    assert composed.recovery._schedule.alpha == 1.0


def test_freeze_global_learning_rate_progress_if_supported_is_noop_for_plain_scheduler():
    """
    freeze_global_learning_rate_progress_if_supported should be a no-op around LearningRateScheduler.
    """
    # Arrange
    scheduler = ActionLearningRateScheduler(ScheduleMode.CONSTANT, alpha=0.01)

    # Act
    with freeze_global_learning_rate_progress_if_supported(scheduler):
        value = scheduler.alpha_scheduler(0, 5)

    # Assert
    assert value == 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
