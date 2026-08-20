"""Unit tests for GRAD-MATCH simulation-set sampling."""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from growingnn.simulation.simulation_sets.grad_match import GradMatchSimulationSet, omp_select


def test_omp_recovers_a_sparse_gradient_combination():
    """
    OMP should recover the three rows that construct the reference gradient.
    """
    # Arrange
    gradients = torch.eye(5)
    g_reference = gradients[0] + 2 * gradients[2] + 3 * gradients[4]

    # Act
    selected, _weights = omp_select(gradients, g_reference, count=3)

    # Assert
    assert set(selected) == {0, 2, 4}


def test_grad_match_keeps_every_class():
    """
    GradMatchSimulationSet should keep the class quota on a tiny labeled set.
    """
    # Arrange
    x = torch.randn(12, 2)
    y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    train = DataLoader(TensorDataset(x, y), batch_size=4)
    val = DataLoader(TensorDataset(x[:6], y[:6]), batch_size=3)
    model = nn.Linear(2, 3)

    # Act
    sim_train, _ = GradMatchSimulationSet().generate(train, val, size=6, seed=0, model=model)
    labels = y[list(sim_train.dataset.indices)]

    # Assert
    assert set(labels.tolist()) == {0, 1, 2}
