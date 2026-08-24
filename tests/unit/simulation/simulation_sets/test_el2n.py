"""Unit tests for EL2N simulation-set sampling."""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from growingnn.simulation.simulation_sets.commons import subset_indices
from growingnn.simulation.simulation_sets.el2n import El2nSimulationSet


def _separable_loaders():
    x = torch.tensor([[10.0, 0.0], [9.0, 0.0], [0.0, 10.0], [0.0, 9.0]])
    y = torch.tensor([0, 1, 1, 1])
    train = DataLoader(TensorDataset(x, y), batch_size=4)
    val = DataLoader(TensorDataset(x[:2], y[:2]), batch_size=2)
    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(2))
    return train, val, model, y


def test_el2n_selects_the_swapped_label():
    """
    Highest EL2N should pick the example whose label disagrees with a confident prediction.
    """
    # Arrange
    train, val, model, y = _separable_loaders()

    # Act
    sim_train, _ = El2nSimulationSet().generate(train, val, size=2, seed=0, model=model)
    picked = subset_indices(sim_train)

    # Assert
    assert 1 in picked


def test_el2n_is_deterministic_for_a_frozen_model():
    """
    EL2N on a frozen model should return the same indices when generate is called twice.
    """
    # Arrange
    train, val, model, _ = _separable_loaders()
    sampler = El2nSimulationSet()

    # Act
    first, _ = sampler.generate(train, val, size=2, seed=0, model=model)
    second, _ = sampler.generate(train, val, size=2, seed=0, model=model)

    # Assert
    assert subset_indices(first) == subset_indices(second)
