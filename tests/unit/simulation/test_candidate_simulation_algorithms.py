"""Unit tests for keep-set simulation algorithms (mocked scoring)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.fx as fx
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from growingnn.core.config import RunningConfig
from growingnn.core.traced_model import TracedModel
import growingnn.simulation.simulation_algorithms.beam_search_alg as beam_search_alg
import growingnn.simulation.simulation_algorithms.best_first_alg as best_first_alg
import growingnn.simulation.simulation_algorithms.sequential_halving_beam_alg as sequential_halving_beam_alg
import growingnn.simulation.simulation_algorithms.ugape_deepen_alg as ugape_deepen_alg

KEEP_SET_ALGS = (
    beam_search_alg,
    best_first_alg,
    sequential_halving_beam_alg,
    ugape_deepen_alg,
)


class _TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, 3, padding=1)
        self.fc = nn.Linear(4 * 8 * 8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv(x))
        x = torch.nn.functional.adaptive_avg_pool2d(x, (8, 8))
        return self.fc(x.flatten(1))


def _running_config(*, simulation_time: float = 0.75) -> RunningConfig:
    x = torch.randn(16, 1, 8, 8)
    y = torch.randint(0, 2, (16,))
    train = DataLoader(TensorDataset(x[:12], y[:12]), batch_size=4, shuffle=True)
    val = DataLoader(TensorDataset(x[12:], y[12:]), batch_size=4)
    cfg = RunningConfig(generations=1, epochs=1, criterion=nn.CrossEntropyLoss())
    cfg.set_simulation_loaders(train, val)
    cfg.simulation_scheduler.simulation_time = simulation_time
    # Fast deterministic grades: avoid real simulation GD in unit tests.
    cfg.simulation_score = SimpleNamespace(
        score=lambda _gm, _cfg: 0.5,
    )
    cfg.ACTIONS_ENABLE_ADD_SEQ_DROPOUT_01 = False
    cfg.ACTIONS_ENABLE_ADD_SEQ_DROPOUT_02 = False
    cfg.ACTIONS_ENABLE_ADD_SEQ_DROPOUT_05 = False
    cfg.ACTIONS_ENABLE_ADD_NEURONS_11 = False
    cfg.ACTIONS_ENABLE_ADD_NEURONS_15 = False
    cfg.ACTIONS_ENABLE_ADD_NEURONS_20 = False
    cfg.ACTIONS_ENABLE_DEL_NEURONS_01 = False
    cfg.ACTIONS_ENABLE_DEL_NEURONS_05 = False
    cfg.ACTIONS_ENABLE_DEL_NEURONS_09 = False
    return cfg


@pytest.mark.parametrize("alg", KEEP_SET_ALGS, ids=[m.__name__.split(".")[-1] for m in KEEP_SET_ALGS])
def test_keep_set_algorithm_returns_executable_action(alg):
    """
    Each keep-set simulation algorithm should return an executable root action under a short budget.
    """
    # Arrange
    gm = fx.symbolic_trace(_TinyNet())
    traced = TracedModel.create(gm, (1, 1, 8, 8))
    cfg = _running_config()

    # Act
    action, max_depth, rollouts = alg.get_action(traced, cfg)

    # Assert
    assert action is not None
    assert rollouts >= 1
    assert max_depth >= 0
    action.execute(traced)


def test_exp005_imports_keep_set_algorithm_variants():
    """
    Exp 005 should list montecarlo, greedy, random, and the four keep-set candidate algorithms.
    """
    # Arrange / Act
    from experiments.train_mnist_exp005_simulation_algorithms import ALG_VARIANTS

    ids = {alg_id for alg_id, _ in ALG_VARIANTS}

    # Assert
    assert "montecarlo" in ids
    assert "greedy" in ids
    assert "random" in ids
    assert "beam_search" in ids
    assert "best_first" in ids
    assert "sequential_halving_beam" in ids
    assert "ugape_deepen" in ids
    assert len(ALG_VARIANTS) == 7


def test_exp005_includes_big_and_medium_starters():
    """
    Exp 005 should run the Exp 003 big and medium starters for every simulation algorithm.
    """
    # Arrange / Act
    from experiments.train_mnist_exp005_simulation_algorithms import MODEL_VARIANTS

    names = [model_name for model_name, _ in MODEL_VARIANTS]

    # Assert
    assert names == ["big", "medium_1conv_2linear"]
    assert len(MODEL_VARIANTS) == 2


def test_exp005_simulation_algorithm_ids_are_unique_and_stable():
    """
    Each Exp 005 keep-set simulation run family should have a unique stable simulation_alg_id.
    """
    # Arrange
    from experiments.train_mnist_exp005_simulation_algorithms import ALG_VARIANTS

    # Act
    ids = [alg_id for alg_id, _ in ALG_VARIANTS]

    # Assert
    assert len(ids) == len(set(ids))
    assert ids == [
        "montecarlo",
        "greedy",
        "random",
        "beam_search",
        "best_first",
        "sequential_halving_beam",
        "ugape_deepen",
    ]
