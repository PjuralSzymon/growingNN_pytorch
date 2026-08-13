"""Unit tests for ``growingnn.utils.seed``."""

import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from growingnn.utils.seed import seed_all


def test_seed_all_makes_python_numpy_and_torch_streams_reproducible():
    """
    seed_all should reset Python, NumPy, and PyTorch RNGs to the same seed.
    """
    # Arrange
    seed = 123

    # Act
    seed_all(seed)
    python_a = random.random()
    numpy_a = float(np.random.random())
    torch_a = float(torch.rand(1).item())

    seed_all(seed)
    python_b = random.random()
    numpy_b = float(np.random.random())
    torch_b = float(torch.rand(1).item())

    # Assert
    assert python_a == python_b
    assert numpy_a == numpy_b
    assert torch_a == pytest.approx(torch_b)
