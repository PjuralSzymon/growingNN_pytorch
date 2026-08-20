"""Unit tests for k-Center CoreSet simulation-set sampling."""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from growingnn.simulation.simulation_sets.commons import subset_indices
from growingnn.simulation.simulation_sets.kcenter import KCenterSimulationSet


def test_kcenter_covers_two_blobs_in_one_class():
    """
    Per-class k-Center should keep points from two well-separated blobs, not only the centroid blob.
    """
    # Arrange
    blob_a = torch.tensor([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]])
    blob_b = torch.tensor([[10.0, 0.0], [10.1, 0.0], [10.0, 0.1]])
    other = torch.tensor([[0.0, 10.0], [0.1, 10.0]])
    x = torch.cat([blob_a, blob_b, other], dim=0)
    y = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1])
    train = DataLoader(TensorDataset(x, y), batch_size=8)
    val = DataLoader(TensorDataset(x[:4], y[:4]), batch_size=4)
    model = nn.Linear(2, 2)

    # Act
    sim_train, _ = KCenterSimulationSet().generate(train, val, size=4, seed=0, model=model)
    picked = subset_indices(sim_train)
    class0 = [i for i in picked if y[i].item() == 0]
    xs = x[class0, 0]

    # Assert
    assert (xs < 1).any()
    assert (xs > 9).any()


def test_kcenter_keeps_every_class():
    """
    KCenterSimulationSet should keep at least one example from every class when size >= n_classes.
    """
    # Arrange
    x = torch.randn(9, 2)
    y = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
    train = DataLoader(TensorDataset(x, y), batch_size=3)
    val = DataLoader(TensorDataset(x[:6], y[:6]), batch_size=3)
    model = nn.Linear(2, 3)

    # Act
    sim_train, _ = KCenterSimulationSet().generate(train, val, size=3, seed=0, model=model)
    labels = torch.cat([batch[1] for batch in sim_train])

    # Assert
    assert set(labels.tolist()) == {0, 1, 2}
