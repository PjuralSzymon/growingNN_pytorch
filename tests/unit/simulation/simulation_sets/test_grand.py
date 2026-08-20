"""Unit tests for GraNd simulation-set sampling."""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from growingnn.simulation.simulation_sets.commons import subset_indices
from growingnn.simulation.simulation_sets.grand import GrandSimulationSet


class _TwoLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(2, 4, bias=False)
        self.head = nn.Linear(4, 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.hidden(x))


def test_grand_selects_the_swapped_label():
    """
    Highest last-layer GraNd should include the example whose label was swapped.
    """
    # Arrange
    x = torch.tensor([[10.0, 0.0], [9.0, 0.0], [0.0, 10.0], [0.0, 9.0]])
    y = torch.tensor([0, 1, 1, 1])
    train = DataLoader(TensorDataset(x, y), batch_size=4)
    val = DataLoader(TensorDataset(x[:2], y[:2]), batch_size=2)
    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(2))

    # Act
    sim_train, _ = GrandSimulationSet().generate(train, val, size=2, seed=0, model=model)
    picked = subset_indices(sim_train)

    # Assert
    assert 1 in picked


def test_grand_uses_last_layer_gradients_only():
    """
    GraNd scoring should not populate gradients on earlier layers.
    """
    # Arrange
    x = torch.randn(6, 2)
    y = torch.tensor([0, 0, 0, 1, 1, 1])
    train = DataLoader(TensorDataset(x, y), batch_size=3)
    val = DataLoader(TensorDataset(x[:3], y[:3]), batch_size=3)
    model = _TwoLayer()

    # Act
    GrandSimulationSet().generate(train, val, size=2, seed=0, model=model)

    # Assert
    assert model.hidden.weight.grad is None
