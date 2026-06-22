"""Unit tests for SimulationScore."""

from __future__ import annotations

import pytest

from growingnn.board.experiment_board import ExperimentBoard
from growingnn.core.config import RunningConfig
from growingnn.simulation.score_functions.simulation_score import SimulationScore


def test_score_fills_experiment_board_simulation_metrics():
    """
    score should write per-term and composite values into experiment_board.simulation_metrics.
    """
    # Arrange
    board = ExperimentBoard("/tmp/unused", experiment_name="t")
    score_fn = SimulationScore(weight_acc=0.5, weight_loss=0.0, weight_time=0.0, weight_countW=0.5)

    def fake_acc(_model, _config):
        return 0.4

    def fake_count(_model, _config):
        return 0.2

    score_fn._SCORE_FUNCTIONS = {"weight_acc": fake_acc, "weight_countW": fake_count}
    score_fn.weights = {"weight_acc": 0.5, "weight_loss": 0.0, "weight_time": 0.0, "weight_countW": 0.5}
    cfg = RunningConfig(
        generations=1,
        epochs=1,
        enable_experiment_board=True,
        experiment_board=board,
        simulation_score=score_fn,
    )

    # Act
    composite = score_fn.score(object(), cfg)

    # Assert
    assert composite == pytest.approx(0.3)
    assert board.simulation_metrics["weight_acc_score"] == 0.4
    assert board.simulation_metrics["weight_acc_weight"] == 0.5
    assert board.simulation_metrics["weight_countW_score"] == 0.2
    assert board.simulation_metrics["composite_score"] == pytest.approx(0.3)


def test_simulation_score_weight_sum():
    """
    weight_sum should return the sum of configured term weights.
    """
    # Arrange
    score_fn = SimulationScore(weight_acc=1.0, weight_loss=0.0, weight_time=0.0, weight_countW=0.5)

    # Act
    total = score_fn.weight_sum()

    # Assert
    assert total == 1.5
