"""Unit tests for moderate-difficulty simulation-set sampling."""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from growingnn.simulation.simulation_sets.commons import subset_indices
from growingnn.simulation.simulation_sets.moderate_difficulty import ModerateDifficultySimulationSet


class _TableNet(nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.head = nn.Linear(1, logits.shape[1])
        self.register_buffer("table", logits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.table[x.view(-1).long()]


def test_moderate_difficulty_skips_easy_and_hard_extremes():
    """
    Middle-50% sampling should not pick the easiest or hardest example of a class.
    """
    # Arrange
    logits = torch.tensor([
        [8.0, 0.0],
        [0.4, 0.0],
        [0.2, 0.0],
        [-8.0, 8.0],
        [8.0, 0.0],
        [0.4, 0.0],
        [0.2, 0.0],
        [-8.0, 8.0],
    ])
    x = torch.arange(8).float().unsqueeze(1)
    y = torch.zeros(8, dtype=torch.long)
    train = DataLoader(TensorDataset(x, y), batch_size=8)
    val = DataLoader(TensorDataset(x[:2], y[:2]), batch_size=2)
    model = _TableNet(logits)

    # Act
    sim_train, _ = ModerateDifficultySimulationSet().generate(train, val, size=1, seed=0, model=model)
    picked = subset_indices(sim_train)

    # Assert
    assert picked[0] not in {0, 3, 4, 7}


def test_moderate_difficulty_keeps_every_class():
    """
    ModerateDifficultySimulationSet should keep the class quota so every class appears.
    """
    # Arrange
    x = torch.randn(12, 2)
    y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    train = DataLoader(TensorDataset(x, y), batch_size=4)
    val = DataLoader(TensorDataset(x[:6], y[:6]), batch_size=3)
    model = nn.Linear(2, 3)

    # Act
    sim_train, _ = ModerateDifficultySimulationSet().generate(train, val, size=6, seed=0, model=model)
    labels = torch.cat([batch[1] for batch in sim_train])

    # Assert
    assert set(labels.tolist()) == {0, 1, 2}
