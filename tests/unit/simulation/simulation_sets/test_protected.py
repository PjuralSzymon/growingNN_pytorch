"""Unit tests for protected stratified simulation-set sampling."""

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from growingnn.simulation.simulation_sets.base import ProtectedSimulationSet
from growingnn.simulation.simulation_sets.commons import protected_sampling_indices


def test_protected_sampling_indices_balances_classes():
    """
    protected_sampling_indices should sample from every class in the label vector.
    """
    # Arrange
    labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])

    # Act
    indices = protected_sampling_indices(labels, n=6, seed=0)
    picked = labels[indices]

    # Assert
    assert len(set(picked.tolist())) == 3


def test_protected_generate_covers_every_class():
    """
    ProtectedSimulationSet.generate should build smaller loaders that still cover every class.
    """
    # Arrange
    x = torch.randn(30, 4)
    y = torch.randint(0, 3, (30,))
    train = DataLoader(TensorDataset(x, y), batch_size=5)
    val = DataLoader(TensorDataset(x[:12], y[:12]), batch_size=4)

    # Act
    sim_train, sim_val = ProtectedSimulationSet().generate(train, val, size=9, seed=0)
    train_labels = torch.cat([batch[1] for batch in sim_train])
    val_labels = torch.cat([batch[1] for batch in sim_val])

    # Assert
    assert len(sim_train.dataset) >= 3
    assert len(torch.unique(train_labels)) == 3
    assert len(val_labels) >= 1
