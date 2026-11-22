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


def test_simulation_score_defaults_to_validation_accuracy_metric():
    """
    SimulationScore should default accuracy_metric to val_acc for backward compatibility.
    """
    # Arrange / Act
    score_fn = SimulationScore()

    # Assert
    assert score_fn.accuracy_metric.value == "val_acc"


def test_simulation_score_records_accuracy_metric_on_board():
    """
    score should store the configured accuracy_metric in board simulation metrics.
    """
    # Arrange
    board = ExperimentBoard("/tmp/unused", experiment_name="t")
    score_fn = SimulationScore(
        weight_acc=1.0,
        weight_loss=0.0,
        weight_time=0.0,
        weight_countW=0.0,
        accuracy_metric="train_acc",
    )
    score_fn._SCORE_FUNCTIONS = {"weight_acc": lambda _model, _config: 0.7}
    score_fn.weights = {
        "weight_acc": 1.0,
        "weight_loss": 0.0,
        "weight_time": 0.0,
        "weight_countW": 0.0,
    }
    cfg = RunningConfig(
        generations=1,
        epochs=1,
        enable_experiment_board=True,
        experiment_board=board,
        simulation_score=score_fn,
    )

    # Act
    score_fn.score(object(), cfg)

    # Assert
    assert board.simulation_metrics["accuracy_metric"] == "train_acc"
    assert board.simulation_metrics["weight_acc_score"] == 0.7


def test_score_and_train_returns_the_copy_trained_by_the_accuracy_term(monkeypatch):
    import torch
    from types import SimpleNamespace
    from growingnn.simulation.score_functions.simulation_score import SimulationScore

    def fake_accuracy(model, cfg):
        with torch.no_grad():
            model.weight.fill_(1.0)  # stands in for scoring training, which updates the copy in place
        return 0.7

    monkeypatch.setitem(SimulationScore._SCORE_FUNCTIONS, 'weight_acc', fake_accuracy)
    model = torch.nn.Linear(2, 2)
    before = model.weight.detach().clone()
    config = SimpleNamespace(experiment_board=None)

    score, trained = SimulationScore(weight_acc=1.0, weight_countW=0.0).score_and_train(model, config)

    assert score == 0.7
    assert torch.equal(trained.weight, torch.ones(2, 2))
    assert torch.equal(model.weight, before)  # the caller's model is never trained
    assert SimulationScore(weight_acc=1.0, weight_countW=0.0).score(model, config) == 0.7


def test_score_and_train_uses_the_time_term_when_it_is_the_only_one_that_trains(monkeypatch):
    import torch
    from types import SimpleNamespace
    from growingnn.simulation.score_functions.simulation_score import SimulationScore

    def fake_time(model, cfg):
        with torch.no_grad():
            model.weight.fill_(2.0)  # the time term trains its copy to measure training time
        return 0.4

    monkeypatch.setitem(SimulationScore._SCORE_FUNCTIONS, 'weight_time', fake_time)
    scorer = SimulationScore(weight_acc=0.0, weight_time=1.0, weight_countW=0.0)

    _, trained = scorer.score_and_train(torch.nn.Linear(2, 2), SimpleNamespace(experiment_board=None))

    assert torch.equal(trained.weight, torch.full((2, 2), 2.0))
