"""Unit tests for Experiment 003 score-metric grid constants and factories."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments import experiments_common as common
from experiments.train_mnist_exp003_score_accuracy_metric import (
    GENERATIONS,
    MODEL_VARIANTS,
    RUNS_DIR,
    SCORE_METRICS,
    SEEDS,
    SIMULATION_TIME_SEC,
    SLOPE_ANGLE_THRESHOLD,
)
from growingnn.simulation.score_functions.score_by_learning import AccuracyMetric
from growingnn.simulation.score_functions.simulation_score import SimulationScore


def test_exp003_uses_four_seeds_and_fixed_three_degree_slope():
    """
    Experiment 003 should reuse Exp 002 seeds and the fixed 3° slope gate.
    """
    # Arrange / Act / Assert
    assert SEEDS == (100, 101, 102, 103)
    assert SLOPE_ANGLE_THRESHOLD == 3.0
    assert GENERATIONS == 5
    assert SIMULATION_TIME_SEC == 120.0


def test_exp003_compares_validation_and_training_score_metrics():
    """
    The grid should sweep validation grading and training grading only.
    """
    # Arrange / Act / Assert
    assert SCORE_METRICS == (AccuracyMetric.VAL_ACC, AccuracyMetric.TRAIN_ACC)
    assert RUNS_DIR.name == "exp003_score_accuracy_metric"


def test_exp003_registers_big_and_medium_1conv_2linear_only():
    """
    Exp 003 should keep only the two strongest corrected Exp 002 starters.
    """
    # Arrange / Act
    names = [name for name, _ in MODEL_VARIANTS]

    # Assert
    assert names == ["big", "medium_1conv_2linear"]


def test_exp003_factories_forward_mnist_batch():
    """
    Every registered factory should accept (N,1,28,28) and return (N,10).
    """
    # Arrange
    probe = torch.randn(2, 1, 28, 28)

    for name, factory in MODEL_VARIANTS:
        # Act
        output = factory({})(probe)

        # Assert
        assert output.shape == (2, 10), name


def test_running_config_forwards_score_accuracy_metric_hyperparameter():
    """
    experiments_common should pass score_accuracy_metric into SimulationScore.
    """
    # Arrange
    hp = {
        "generations": 1,
        "epochs": 1,
        "lr_alpha": 0.01,
        "simulation_time": 1.0,
        "simulation_epochs": 1,
        "target_accuracy": 1.0,
        "score_weight_acc": 1.0,
        "score_weight_countw": 0.0,
        "simulation_set_size": 8,
        "score_accuracy_metric": "train_acc",
    }

    # Act
    with patch.object(common, "montecarlo_alg", object()):
        config = common._running_config(hp, torch.device("cpu"), board=None)

    # Assert
    assert isinstance(config.simulation_score, SimulationScore)
    assert config.simulation_score.accuracy_metric is AccuracyMetric.TRAIN_ACC


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
