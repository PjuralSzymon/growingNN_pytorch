"""Unit tests for Monte Carlo tree search simulation."""

import asyncio
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import growingnn.core.config

growingnn.core.config.ENABLE_LOGGING = False

from growingnn.simulation.simulation_algorithms import montecarlo_alg


def test_get_action_logs_error_when_rollouts_stall_after_deadline(caplog):
    """
    get_action should log an error when past deadline and _simulate returns
    the same rollout count twice in a row.
    """
    # Arrange
    caplog.set_level(logging.ERROR, logger="growingnn")
    model = MagicMock()
    running_config = MagicMock()
    running_config.simulation_scheduler.simulation_time = 0.0
    actions = [MagicMock(), MagicMock()]
    simulate_results = [(0.0, 0, 1), (0.0, 0, 1), (0.0, 0, 3)]

    with (
        patch.object(montecarlo_alg, "generate_all_actions", return_value=actions),
        patch.object(montecarlo_alg, "time") as mock_time,
        patch.object(montecarlo_alg, "_simulate", side_effect=simulate_results),
        patch.object(montecarlo_alg, "clear_reshepers_cache"),
    ):
        past_deadline = iter([0.0] + [100.0] * 20)
        mock_time.time.side_effect = lambda: next(past_deadline)

        # Act
        _, _, rollouts = asyncio.run(montecarlo_alg.get_action(model, running_config))

    # Assert
    assert rollouts == 3
    assert sum("no new rollouts after deadline" in record.message for record in caplog.records) == 1


def test_get_action_breaks_when_rollouts_exceed_action_count_after_deadline():
    """
    get_action should stop once rollouts exceed the root action count after deadline.
    """
    # Arrange
    model = MagicMock()
    running_config = MagicMock()
    running_config.simulation_scheduler.simulation_time = 0.0
    actions = [MagicMock(), MagicMock()]

    with (
        patch.object(montecarlo_alg, "generate_all_actions", return_value=actions),
        patch.object(montecarlo_alg, "time") as mock_time,
        patch.object(montecarlo_alg, "_simulate", return_value=(0.0, 0, 3)),
        patch.object(montecarlo_alg, "clear_reshepers_cache"),
    ):
        past_deadline = iter([0.0] + [100.0] * 10)
        mock_time.time.side_effect = lambda: next(past_deadline)

        # Act
        _, _, rollouts = asyncio.run(montecarlo_alg.get_action(model, running_config))

    # Assert
    assert rollouts == 3
    montecarlo_alg._simulate.assert_called_once()
