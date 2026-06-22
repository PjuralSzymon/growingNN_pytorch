"""Unit tests for simulation scheduling."""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from growingnn.simulation.simulation_scheduler import SchedulerMode, SimulationScheduler


def test_progress_check_triggers_on_stagnation():
    """
    PROGRESS_CHECK should simulate when validation accuracy stops improving.
    """
    # Arrange
    scheduler = SimulationScheduler(SchedulerMode.PROGRESS_CHECK, stagnation_window=1)

    # Act
    result = scheduler.can_simulate(1, [0.4, 0.4])

    # Assert
    assert result is True


def test_never_mode_skips_simulation():
    """
    NEVER mode should never trigger simulation.
    """
    # Arrange
    scheduler = SimulationScheduler(SchedulerMode.NEVER)

    # Act
    result = scheduler.can_simulate(1, [0.2, 0.5])

    # Assert
    assert result is False
