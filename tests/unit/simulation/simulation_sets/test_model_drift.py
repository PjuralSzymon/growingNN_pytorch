"""Unit tests for model-drift simulation-set refresh."""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from growingnn.simulation.simulation_sets.base import SimulationSet
from growingnn.simulation.simulation_sets.protected import ProtectedSimulationSet
from growingnn.simulation.simulation_sets.commons import subset_indices
from growingnn.simulation.simulation_sets.model_drift import ModelDriftSimulationSet


class _TwoLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(2, 4)
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.relu(self.hidden(x)))


class _CountingSelector(SimulationSet):
    def __init__(self) -> None:
        self.calls = 0
        self.inner = ProtectedSimulationSet()

    def generate(self, train_loader, val_loader, size, seed=0, model=None):
        self.calls += 1
        return self.inner.generate(train_loader, val_loader, size, seed, model)


def _loaders():
    x = torch.randn(20, 2)
    y = torch.randint(0, 2, (20,))
    train = DataLoader(TensorDataset(x, y), batch_size=5)
    val = DataLoader(TensorDataset(x[:8], y[:8]), batch_size=4)
    return train, val


def test_model_drift_reuses_indices_when_embeddings_are_stable():
    """
    A second generate call on an unchanged model should keep the cached simulation indices.
    """
    # Arrange
    train, val = _loaders()
    model = _TwoLayer()
    inner = _CountingSelector()
    sampler = ModelDriftSimulationSet(selector=inner, anchor_size=8, drift_threshold=0.1)

    # Act
    first, _ = sampler.generate(train, val, size=4, seed=0, model=model)
    second, _ = sampler.generate(train, val, size=4, seed=0, model=model)

    # Assert
    assert inner.calls == 1
    assert subset_indices(first) == subset_indices(second)


def test_model_drift_rebuilds_after_embedding_shift():
    """
    Drift above the threshold should call the inner selector again.
    """
    # Arrange
    train, val = _loaders()
    model = _TwoLayer()
    inner = _CountingSelector()
    sampler = ModelDriftSimulationSet(selector=inner, anchor_size=8, drift_threshold=0.05)
    sampler.generate(train, val, size=4, seed=0, model=model)

    # Act
    with torch.no_grad():
        model.hidden.bias.add_(8.0)
    sampler.generate(train, val, size=4, seed=0, model=model)

    # Assert
    assert inner.calls == 2
