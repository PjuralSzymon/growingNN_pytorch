"""Unit tests for RunningConfig."""

from __future__ import annotations

from growingnn.core.config import RunningConfig
from growingnn.training.lr_scheduler_action import ActionLearningRateScheduler, LearningRateScheduler, ScheduleMode


def test_enable_experiment_board_false_clears_board_instance():
    """
    When enable_experiment_board is False, experiment_board must stay None even if passed.
    """
    # Arrange
    sentinel = object()

    # Act
    cfg = RunningConfig(
        generations=1,
        epochs=1,
        lr_scheduler=ActionLearningRateScheduler(ScheduleMode.CONSTANT, alpha=0.01),
        enable_experiment_board=False,
        experiment_board=sentinel,
    )

    # Assert
    assert cfg.enable_experiment_board is False
    assert cfg.experiment_board is None


def test_enable_experiment_board_true_keeps_board_instance():
    """
    When enable_experiment_board is True, the provided experiment_board is kept.
    """
    # Arrange
    sentinel = object()

    # Act
    cfg = RunningConfig(
        generations=1,
        epochs=1,
        enable_experiment_board=True,
        experiment_board=sentinel,
    )

    # Assert
    assert cfg.enable_experiment_board is True
    assert cfg.experiment_board is sentinel


def test_running_config_device_defaults_to_cpu_or_cuda():
    """
    RunningConfig.device should resolve to cuda when available, else cpu.
    """
    # Arrange / Act
    cfg = RunningConfig(generations=1, epochs=1)

    # Assert
    assert cfg.device in ("cpu", "cuda")
