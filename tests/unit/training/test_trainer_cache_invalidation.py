"""Unit tests for TracedModel cache invalidation inside train_generations."""

import sys
import types
from pathlib import Path
from unittest.mock import patch

import torch
import torch.fx as fx
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import growingnn.core.config

growingnn.core.config.ENABLE_LOGGING = False

from growingnn.actions.add_seq_linear_layer import AddSeqLinearLayer
from growingnn.core.config import RunningConfig
from growingnn.core.traced_model import TracedModel
from growingnn.simulation.simulation_schedulers import (
    AlwaysSimulationScheduler,
    NeverSimulationScheduler,
)
from growingnn.training.lr_scheduler import LearningRateScheduler, ScheduleMode
from growingnn.training.stoppers import StopperMode, TrainingStopper
from growingnn.training.trainer import train_generations
from growingnn.utils.fx import GraphStructureQuery
from tests.model_factory import ModelFactory


def _linear_loaders(batch_size: int = 4, n: int = 16):
    torch.manual_seed(0)
    x = torch.randn(n, 4)
    y = torch.randint(0, 2, (n,))
    train = DataLoader(TensorDataset(x[:12], y[:12]), batch_size=batch_size, shuffle=True)
    val = DataLoader(TensorDataset(x[12:], y[12:]), batch_size=batch_size)
    return train, val


def _grow_only_config(*, generations: int = 3) -> RunningConfig:
    cfg = RunningConfig(
        generations=generations,
        epochs=1,
        lr_scheduler=LearningRateScheduler(ScheduleMode.CONSTANT, alpha=0.01),
        stopper=TrainingStopper(StopperMode.EMPTY),
        simulation_scheduler=AlwaysSimulationScheduler(simulation_time=0.01),
        simulation_set_size=8,
        criterion=nn.CrossEntropyLoss(),
        quiet=True,
    )
    cfg.update_shrink_actions(False)
    cfg.update_grow_actions(False)
    cfg.ACTIONS_ENABLE_ADD_SEQ_LAYER = True
    return cfg


def _deterministic_add_seq_linear(traced: TracedModel, running_config):
    actions = AddSeqLinearLayer.generate_all_actions(traced)
    if not actions:
        return None, 0, 0
    return actions[0], 0, 1


def _deterministic_sim_alg():
    return types.SimpleNamespace(get_action=_deterministic_add_seq_linear)


def _chain_model():
    return fx.symbolic_trace(ModelFactory.simple_chain_3())


def test_train_generations_invalidates_after_each_simulation_action():
    """
    train_generations should call TracedModel.invalidate once per executed simulation action.
    """
    # Arrange
    gm = _chain_model()
    train_loader, val_loader = _linear_loaders()
    cfg = _grow_only_config(generations=3)
    cfg.simulation_alg = _deterministic_sim_alg()
    invalidate_calls: list[int] = []
    original_invalidate = TracedModel.invalidate

    def counting_invalidate(self) -> None:
        invalidate_calls.append(1)
        original_invalidate(self)

    # Act
    with patch.object(TracedModel, "invalidate", counting_invalidate):
        train_generations(gm, train_loader, val_loader, cfg)

    # Assert
    assert len(invalidate_calls) == 3


def test_train_generations_skips_invalidate_when_simulation_disabled():
    """
    train_generations should not invalidate TracedModel caches when simulation never runs.
    """
    # Arrange
    gm = _chain_model()
    train_loader, val_loader = _linear_loaders()
    cfg = _grow_only_config(generations=2)
    cfg.simulation_scheduler = NeverSimulationScheduler()
    invalidate_calls: list[int] = []
    original_invalidate = TracedModel.invalidate

    def counting_invalidate(self) -> None:
        invalidate_calls.append(1)
        original_invalidate(self)

    # Act
    with patch.object(TracedModel, "invalidate", counting_invalidate):
        train_generations(gm, train_loader, val_loader, cfg)

    # Assert
    assert invalidate_calls == []


def test_train_generations_recomputes_param_count_after_each_invalidate():
    """
    param_count() during training should re-query graph size after each post-simulation invalidate.
    """
    # Arrange
    gm = _chain_model()
    train_loader, val_loader = _linear_loaders()
    cfg = _grow_only_config(generations=3)
    cfg.simulation_alg = _deterministic_sim_alg()
    params_before = GraphStructureQuery.get_amount_of_parameters(gm)
    recomputed_counts: list[int] = []
    original_count = GraphStructureQuery.get_amount_of_parameters

    def counting_param_count(gm):
        count = original_count(gm)
        recomputed_counts.append(count)
        return count

    # Act
    with patch.object(GraphStructureQuery, "get_amount_of_parameters", counting_param_count):
        train_generations(gm, train_loader, val_loader, cfg)

    # Assert
    assert recomputed_counts == [params_before, params_before + 20, params_before + 40]
    assert GraphStructureQuery.get_amount_of_parameters(gm) == params_before + 60


def test_train_generations_runs_back_to_back_with_fresh_cache_each_run():
    """
    Repeated train_generations calls should keep growing the model without stale cached analysis.
    """
    # Arrange
    gm = _chain_model()
    train_loader, val_loader = _linear_loaders()
    cfg = _grow_only_config(generations=2)
    cfg.simulation_alg = _deterministic_sim_alg()
    params_before = GraphStructureQuery.get_amount_of_parameters(gm)

    # Act
    gm, _ = train_generations(gm, train_loader, val_loader, cfg)
    params_after_first_run = GraphStructureQuery.get_amount_of_parameters(gm)
    gm, _ = train_generations(gm, train_loader, val_loader, cfg)
    params_after_second_run = GraphStructureQuery.get_amount_of_parameters(gm)

    # Assert
    assert params_after_first_run == params_before + 40
    assert params_after_second_run == params_before + 80
