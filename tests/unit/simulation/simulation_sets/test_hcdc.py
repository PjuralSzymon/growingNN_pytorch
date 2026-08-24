"""Unit tests for HCDC synthetic simulation-set construction."""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from growingnn.simulation.simulation_sets.hcdc import HcdcSimulationSet


def _loaders():
    x = torch.randn(16, 2)
    y = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
    train = DataLoader(TensorDataset(x, y), batch_size=4)
    val = DataLoader(TensorDataset(x[:8], y[:8]), batch_size=4)
    model = nn.Linear(2, 2)
    return train, val, model


def test_hcdc_returns_requested_train_size_and_labels():
    """
    HCDC generate should return a train loader of the requested size with valid class labels.
    """
    # Arrange
    train, val, model = _loaders()

    # Act
    sim_train, _ = HcdcSimulationSet(steps=1, time_cap=5.0).generate(
        train, val, size=4, seed=0, model=model,
    )
    labels = torch.cat([batch[1] for batch in sim_train])

    # Assert
    assert len(sim_train.dataset) == 4
    assert set(labels.tolist()).issubset({0, 1})


def test_hcdc_updates_synthetic_inputs_in_one_step():
    """
    One condensation step should keep finite synthetic inputs that are not stuck at zero.
    """
    # Arrange
    train, val, model = _loaders()
    sampler = HcdcSimulationSet(steps=1, time_cap=5.0)

    # Act
    sim_train, _ = sampler.generate(train, val, size=4, seed=0, model=model)
    data = torch.stack([sim_train.dataset[i][0] for i in range(len(sim_train.dataset))])

    # Assert
    assert torch.isfinite(data).all()
    assert not torch.allclose(data, torch.zeros_like(data))
