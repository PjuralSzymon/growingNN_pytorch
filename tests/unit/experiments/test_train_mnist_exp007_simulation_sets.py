"""Unit tests for Experiment 007 simulation-set grid constants."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.train_mnist_exp007_simulation_sets import (
    SET_VARIANTS,
    SIMULATION_SET_SIZES,
    exp007_folder_name,
)


def test_exp007_keeps_six_generators_and_drops_weak_ones():
    """
    Experiment 007 should keep the six stronger generators and drop grand, grad_match, and hcdc.
    """

    # Arrange / Act
    names = [name for name, _ in SET_VARIANTS]

    # Assert
    assert names == [
        "protected",
        "moderate_difficulty",
        "kcenter",
        "el2n",
        "craig",
        "model_drift",
    ]


def test_exp007_uses_three_small_simulation_set_sizes():
    """
    Experiment 007 should score on sizes 100, 500, and 1000 instead of 2000.
    """

    # Arrange / Act / Assert
    assert SIMULATION_SET_SIZES == (100, 500, 1000)


def test_exp007_folder_name_appends_simulation_set_size():
    """
    Folder names for different simulation set sizes should not collide.
    """

    # Arrange
    hp = {
        "dataset": "mnist",
        "generations": 10,
        "epochs": 10,
        "batch_size": 64,
        "lr_alpha": 0.01,
        "score_weight_countw": 0.1,
        "model_channels": 4,
        "hidden_linear_size": 16,
        "simulation_set_size": 100,
    }

    # Act
    folder_100 = exp007_folder_name(hp)
    folder_500 = exp007_folder_name({**hp, "simulation_set_size": 500})

    # Assert
    assert folder_100.endswith("_simsz100")
    assert folder_500.endswith("_simsz500")
    assert folder_100 != folder_500
