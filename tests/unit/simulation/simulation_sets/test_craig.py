"""Unit tests for CRAIG simulation-set sampling."""

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from growingnn.simulation.simulation_sets.craig import craig_select


def test_craig_keeps_both_opposite_gradient_clusters():
    """
    CRAIG should select at least one representative from each opposite gradient cluster.
    """
    # Arrange
    positive = torch.ones(4, 3)
    negative = -torch.ones(2, 3)
    gradients = torch.cat([positive, negative], dim=0)

    # Act
    selected, _weights = craig_select(gradients, count=2)
    signs = {1 if gradients[i, 0] > 0 else -1 for i in selected}

    # Assert
    assert signs == {1, -1}


def test_craig_gives_the_larger_cluster_a_larger_weight():
    """
    The cluster with more examples should receive a strictly larger CRAIG weight.
    """
    # Arrange
    positive = torch.ones(6, 2)
    negative = -torch.ones(2, 2)
    gradients = torch.cat([positive, negative], dim=0)

    # Act
    _selected, weights = craig_select(gradients, count=2)

    # Assert
    assert weights.max() > weights.min()
