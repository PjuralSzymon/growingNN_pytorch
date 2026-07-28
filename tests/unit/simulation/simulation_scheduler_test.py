"""Unit tests for simulation scheduling."""

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from growingnn.simulation.simulation_schedulers import (
    AlwaysSimulationScheduler,
    MeanStandardDeviationStagnationSimulationScheduler,
    NeverSimulationScheduler,
    ProgressCheckSimulationScheduler,
    SimulationScheduler,
    SlopeEstimationSimulationScheduler,
)
from growingnn.simulation.simulation_schedulers.base import least_squares_slope


def test_base_scheduler_cannot_be_created_directly():
    """
    SimulationScheduler should require a concrete scheduling policy.
    """
    # Arrange / Act / Assert
    with pytest.raises(TypeError):
        SimulationScheduler()


def test_always_mode_permits_simulation():
    """
    AlwaysSimulationScheduler should permit simulation for every metric history.
    """
    # Arrange
    scheduler = AlwaysSimulationScheduler()

    # Act
    result = scheduler.can_simulate(0, [])

    # Assert
    assert result is True


def test_progress_check_triggers_on_stagnation():
    """
    PROGRESS_CHECK should simulate when validation accuracy stops improving.
    """
    # Arrange
    scheduler = ProgressCheckSimulationScheduler(stagnation_window=1)

    # Act
    result = scheduler.can_simulate(1, [0.4, 0.4])

    # Assert
    assert result is True


def test_never_mode_skips_simulation():
    """
    NEVER mode should never trigger simulation.
    """
    # Arrange
    scheduler = NeverSimulationScheduler()

    # Act
    result = scheduler.can_simulate(1, [0.2, 0.5])

    # Assert
    assert result is False


def test_least_squares_slope_returns_linear_rate():
    """
    least_squares_slope should return the per-step rate of a linear sequence.
    """
    # Arrange
    values = [0.1, 0.2, 0.3, 0.4]

    # Act
    result = least_squares_slope(values)

    # Assert
    assert result == pytest.approx(0.1)


def test_slope_estimation_triggers_on_flat_generation_training_accuracy():
    """
    SLOPE_ESTIMATION should simulate when current-generation training accuracy is flat.
    """
    # Arrange
    scheduler = SlopeEstimationSimulationScheduler(
        angle_threshold=1.0,
    )

    # Act
    result = scheduler.can_simulate(
        3,
        [0.91, 0.91, 0.91, 0.91],
    )

    # Assert
    assert result is True


def test_slope_estimation_skips_rising_generation_training_accuracy():
    """
    SLOPE_ESTIMATION should not simulate while current-generation training accuracy is rising.
    """
    # Arrange
    scheduler = SlopeEstimationSimulationScheduler(
        angle_threshold=1.0,
    )

    # Act
    result = scheduler.can_simulate(
        3,
        [0.70, 0.75, 0.80, 0.85],
    )

    # Assert
    assert result is False


def test_slope_estimation_skips_strongly_falling_generation_training_accuracy():
    """
    SLOPE_ESTIMATION should apply its absolute-angle rule to falling training accuracy.
    """
    # Arrange
    scheduler = SlopeEstimationSimulationScheduler(
        angle_threshold=1.0,
    )

    # Act
    result = scheduler.can_simulate(
        3,
        [0.85, 0.80, 0.75, 0.70],
    )

    # Assert
    assert result is False


def test_slope_estimation_requires_two_generations():
    """
    SLOPE_ESTIMATION should not simulate with fewer than two generation accuracies.
    """
    # Arrange
    scheduler = SlopeEstimationSimulationScheduler()

    # Act
    result = scheduler.can_simulate(
        0,
        [0.80],
    )

    # Assert
    assert result is False


def test_slope_estimation_rejects_non_finite_accuracy():
    """
    SLOPE_ESTIMATION should not simulate when its generation history contains NaN.
    """
    # Arrange
    scheduler = SlopeEstimationSimulationScheduler()

    # Act
    result = scheduler.can_simulate(
        3,
        [0.80, 0.81, float("nan"), 0.82],
    )

    # Assert
    assert result is False


def test_mean_standard_deviation_stagnation_triggers_on_constant_accuracy():
    """
    Mean and standard deviation mode should simulate when accuracy has zero deviation.
    """
    # Arrange
    scheduler = MeanStandardDeviationStagnationSimulationScheduler()

    # Act
    result = scheduler.can_simulate(
        3,
        [0.91, 0.91, 0.91, 0.91],
    )

    # Assert
    assert result is True


def test_mean_standard_deviation_stagnation_triggers_on_flat_noise():
    """
    Mean and standard deviation mode should simulate for flat statistical noise.
    """
    # Arrange
    scheduler = MeanStandardDeviationStagnationSimulationScheduler(
        slope_epsilon=1e-4,
        standard_deviation_multiplier=1.5,
    )

    # Act
    result = scheduler.can_simulate(
        5,
        [0.910, 0.912, 0.911, 0.911, 0.912, 0.910],
    )

    # Assert
    assert result is True


def test_mean_standard_deviation_stagnation_skips_rising_accuracy():
    """
    Mean and standard deviation mode should reject a rising accuracy sequence.
    """
    # Arrange
    scheduler = MeanStandardDeviationStagnationSimulationScheduler(
        slope_epsilon=1e-4,
        standard_deviation_multiplier=2.0,
    )

    # Act
    result = scheduler.can_simulate(
        5,
        [0.70, 0.72, 0.74, 0.76, 0.78, 0.80],
    )

    # Assert
    assert result is False


@pytest.mark.parametrize(
    ("scheduler_type", "kwargs"),
    [
        (SlopeEstimationSimulationScheduler, {"angle_threshold": -0.1}),
        (
            MeanStandardDeviationStagnationSimulationScheduler,
            {"standard_deviation_multiplier": 0.0},
        ),
    ],
)
def test_scheduler_rejects_invalid_stagnation_parameters(scheduler_type, kwargs):
    """
    SimulationScheduler should reject invalid stagnation parameter values.
    """
    # Arrange
    scheduler = scheduler_type

    # Act / Assert
    with pytest.raises(ValueError):
        scheduler(**kwargs)
