"""Unit tests for keep-set simulation algorithms (mocked scoring)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.fx as fx
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from growingnn.actions.action import Action
from growingnn.actions.registry import generate_all_actions
import growingnn.core.config as project_config
from growingnn.core.config import RunningConfig
from growingnn.core.traced_model import TracedModel
import growingnn.simulation.simulation_algorithms.beam_search_alg as beam_search_alg
import growingnn.simulation.simulation_algorithms.best_first_alg as best_first_alg
import growingnn.simulation.simulation_algorithms.greedy_alg as greedy_alg
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


class _StubAction(Action):
    def __init__(self, name: str, score: float) -> None:
        super().__init__([name])
        self.name = name
        self.fixed_score = score

    def _execute(self, traced: TracedModel) -> None:
        setattr(traced.gm, "_stub_score", self.fixed_score)

    def __str__(self) -> str:
        return self.name


def _stub_score(gm, _cfg) -> float:
    return float(getattr(gm, "_stub_score", 0.0))


def _running_config(*, simulation_time: float = 0.75, score_fn=None) -> RunningConfig:
    x = torch.randn(16, 1, 8, 8)
    y = torch.randint(0, 2, (16,))
    train = DataLoader(TensorDataset(x[:12], y[:12]), batch_size=4, shuffle=True)
    val = DataLoader(TensorDataset(x[12:], y[12:]), batch_size=4)
    cfg = RunningConfig(generations=1, epochs=1, criterion=nn.CrossEntropyLoss())
    cfg.set_simulation_loaders(train, val)
    cfg.simulation_scheduler.simulation_time = simulation_time
    # Fast deterministic grades: avoid real simulation GD in unit tests.
    cfg.simulation_score = SimpleNamespace(
        score=score_fn if score_fn is not None else (lambda _gm, _cfg: 0.5),
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


@pytest.mark.parametrize("alg", KEEP_SET_ALGS, ids=[m.__name__.split(".")[-1] for m in KEEP_SET_ALGS])
def test_keep_set_algorithm_grades_every_root_action_even_with_zero_budget(alg):
    """
    Each keep-set alg should grade every legal root action once before timed exploration.
    """
    # Arrange
    gm = fx.symbolic_trace(_TinyNet())
    traced = TracedModel.create(gm, (1, 1, 8, 8))
    score_calls = {"n": 0}

    def counting_score(_gm, _cfg):
        score_calls["n"] += 1
        return 0.5

    cfg = _running_config(simulation_time=0.0, score_fn=counting_score)
    n_root = len(generate_all_actions(traced, cfg))
    assert n_root > 0

    # Act
    action, _max_depth, rollouts = alg.get_action(traced, cfg)

    # Assert
    assert action is not None
    assert score_calls["n"] >= n_root
    assert rollouts >= n_root
    action.execute(traced)


def test_simulation_min_algorithm_iteration_runs_is_three():
    """
    Scoring simulation algorithms should share SIMULATION_MIN_ALGORITHM_ITERATION_RUNS = 3.
    """
    # Arrange / Act / Assert
    assert project_config.SIMULATION_MIN_ALGORITHM_ITERATION_RUNS == 3


def test_greedy_scores_every_root_action_even_with_zero_budget():
    """
    greedy_alg should score every remaining root action even when simulation_time is 0.
    """
    # Arrange
    gm = fx.symbolic_trace(_TinyNet())
    traced = TracedModel.create(gm, (1, 1, 8, 8))
    arms = [_StubAction(f"arm_{i}", score=0.1 * (i + 1)) for i in range(4)]
    score_calls = {"n": 0}

    def counting_score(child_gm, cfg):
        score_calls["n"] += 1
        return _stub_score(child_gm, cfg)

    cfg = _running_config(simulation_time=0.0, score_fn=counting_score)

    with patch.object(greedy_alg, "generate_all_actions", return_value=arms):
        # Act
        action, _max_depth, rollouts = greedy_alg.get_action(traced, cfg)

    # Assert
    assert action is arms[-1]
    assert score_calls["n"] == len(arms)
    assert rollouts == len(arms)


def test_sequential_halving_beam_runs_min_rounds_and_picks_late_high_score():
    """
    With an expired budget, Sequential Halving Beam should run three halving
    rounds and choose the last root arm when it has the highest score.
    """
    # Arrange
    gm = fx.symbolic_trace(_TinyNet())
    traced = TracedModel.create(gm, (1, 1, 8, 8))
    n_arms = 8
    arms = [_StubAction(f"arm_{i}", score=0.1 * (i + 1)) for i in range(n_arms)]
    score_calls = {"n": 0}
    min_runs = project_config.SIMULATION_MIN_ALGORITHM_ITERATION_RUNS

    def counting_score(child_gm, cfg):
        score_calls["n"] += 1
        return _stub_score(child_gm, cfg)

    cfg = _running_config(simulation_time=0.0, score_fn=counting_score)
    # First pass n, then living 8 -> 4 -> 2 across three full rounds.
    expected_scores = n_arms + n_arms + (n_arms // 2) + (n_arms // 4)

    def fake_generate(model, _cfg):
        if model is traced:
            return list(arms)
        return []

    with patch.object(sequential_halving_beam_alg, "generate_all_actions", side_effect=fake_generate):
        # Act
        action, _max_depth, rollouts = sequential_halving_beam_alg.get_action(traced, cfg)

    # Assert
    assert min_runs == 3
    assert action is arms[-1]
    assert action is not max(arms[: sequential_halving_beam_alg.BEAM_WIDTH], key=lambda a: a.fixed_score)
    assert score_calls["n"] == expected_scores
    assert rollouts == expected_scores


def test_ugape_deepen_runs_min_extra_pulls_after_expired_time():
    """
    ugape_deepen should make at least three extra pulls after the first pass
    even when the wall-clock budget is already gone.
    """
    # Arrange
    gm = fx.symbolic_trace(_TinyNet())
    traced = TracedModel.create(gm, (1, 1, 8, 8))
    arms = [_StubAction(f"arm_{i}", score=0.1 * (i + 1)) for i in range(4)]
    score_calls = {"n": 0}
    min_runs = project_config.SIMULATION_MIN_ALGORITHM_ITERATION_RUNS

    def counting_score(child_gm, cfg):
        score_calls["n"] += 1
        return _stub_score(child_gm, cfg)

    cfg = _running_config(simulation_time=0.0, score_fn=counting_score)

    with patch.object(ugape_deepen_alg, "generate_all_actions", return_value=arms):
        # Act
        action, _max_depth, rollouts = ugape_deepen_alg.get_action(traced, cfg)

    # Assert
    assert action is not None
    assert score_calls["n"] >= len(arms) + min_runs
    assert rollouts >= len(arms) + min_runs


def test_best_first_runs_min_expansions_after_expired_time():
    """
    best_first should expand at least three nodes after the first pass even
    when the wall-clock budget is already gone.
    """
    # Arrange
    gm = fx.symbolic_trace(_TinyNet())
    traced = TracedModel.create(gm, (1, 1, 8, 8))
    arms = [_StubAction(f"arm_{i}", score=0.1 * (i + 1)) for i in range(4)]
    score_calls = {"n": 0}
    min_runs = project_config.SIMULATION_MIN_ALGORITHM_ITERATION_RUNS

    def counting_score(child_gm, cfg):
        score_calls["n"] += 1
        return _stub_score(child_gm, cfg)

    cfg = _running_config(simulation_time=0.0, score_fn=counting_score)

    with patch.object(best_first_alg, "generate_all_actions", return_value=arms):
        # Act
        action, _max_depth, rollouts = best_first_alg.get_action(traced, cfg)

    # Assert
    assert action is not None
    assert score_calls["n"] >= len(arms) + min_runs
    assert rollouts >= len(arms) + min_runs


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
