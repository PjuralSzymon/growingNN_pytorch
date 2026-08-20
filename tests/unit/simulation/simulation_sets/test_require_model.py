"""Unit tests for missing-model errors in simulation-set generators."""

import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from growingnn.simulation.simulation_sets.commons import require_model
from growingnn.simulation.simulation_sets.craig import CraigSimulationSet
from growingnn.simulation.simulation_sets.el2n import El2nSimulationSet
from growingnn.simulation.simulation_sets.grad_match import GradMatchSimulationSet
from growingnn.simulation.simulation_sets.grand import GrandSimulationSet
from growingnn.simulation.simulation_sets.hcdc import HcdcSimulationSet
from growingnn.simulation.simulation_sets.kcenter import KCenterSimulationSet
from growingnn.simulation.simulation_sets.model_drift import ModelDriftSimulationSet
from growingnn.simulation.simulation_sets.moderate_difficulty import ModerateDifficultySimulationSet


MODEL_AWARE_SETS = [
    ModerateDifficultySimulationSet,
    ModelDriftSimulationSet,
    GradMatchSimulationSet,
    HcdcSimulationSet,
    KCenterSimulationSet,
    GrandSimulationSet,
    El2nSimulationSet,
    CraigSimulationSet,
]


def test_require_model_raises_when_missing():
    """
    require_model should raise ValueError when the model is None.
    """

    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="generate requires a model"):
        require_model(None)


def test_require_model_returns_the_given_model():
    """
    require_model should return the same object when a model is present.
    """

    # Arrange
    model = torch.nn.Linear(2, 2)

    # Act
    result = require_model(model)

    # Assert
    assert result is model


@pytest.mark.parametrize("set_cls", MODEL_AWARE_SETS)
def test_model_aware_generate_raises_without_model(set_cls):
    """
    Model-aware generate should raise when the model is missing instead of using ProtectedSimulationSet.
    """

    # Arrange
    x = torch.randn(8, 2)
    y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    train = DataLoader(TensorDataset(x, y), batch_size=4)
    val = DataLoader(TensorDataset(x[:4], y[:4]), batch_size=4)

    # Act / Assert
    with pytest.raises(ValueError, match="generate requires a model"):
        set_cls().generate(train, val, size=4, seed=0, model=None)
