"""Integration tests for generation-based trainer orchestration."""

import sys
from pathlib import Path

import torch
import torch.fx as fx
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet18

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import growingnn.core.config

growingnn.core.config.ENABLE_LOGGING = False

from growingnn.core.config import RunningConfig
from growingnn.simulation.score_functions.simulation_score import SimulationScore
import growingnn.simulation.simulation_algorithms.random_alg as random_alg
from growingnn.simulation.simulation_schedulers import AlwaysSimulationScheduler
from growingnn.training.lr_scheduler_action import ActionLearningRateScheduler, LearningRateScheduler, ScheduleMode
from growingnn.training.stoppers import StopperMode, TrainingStopper
from growingnn.training.trainer import train_generations
from growingnn.utils.fx import GraphStructureQuery, extract_graph


def _loaders():
    x = torch.randn(32, 3, 32, 32)
    y = torch.randint(0, 2, (32,))
    train = DataLoader(TensorDataset(x[:24], y[:24]), batch_size=8, shuffle=True)
    val = DataLoader(TensorDataset(x[24:], y[24:]), batch_size=8)
    return train, val


def test_train_generations_runs_simulation_between_generations():
    """
    train_generations should train, simulate, mutate, and record per-generation metrics.
    """
    # Arrange
    torch.manual_seed(0)
    gm = extract_graph(resnet18(weights=None, num_classes=2))
    train_loader, val_loader = _loaders()
    params_before = GraphStructureQuery.get_amount_of_parameters(gm)
    cfg = RunningConfig(
        generations=2,
        epochs=1,
        lr_scheduler=ActionLearningRateScheduler(ScheduleMode.CONSTANT, alpha=0.01),
        stopper=TrainingStopper(StopperMode.EMPTY),
        simulation_alg=random_alg,
        simulation_scheduler=AlwaysSimulationScheduler(
            simulation_time=1.0,
            simulation_epochs=1,
        ),
        simulation_score=SimulationScore(weight_acc=0.0, weight_countW=1.0),
        simulation_set_size=16,
        criterion=nn.CrossEntropyLoss(),
        quiet=True,
    )
    cfg.ACTIONS_ENABLE_ADD_SEQ_DROPOUT_01 = False
    cfg.ACTIONS_ENABLE_ADD_SEQ_DROPOUT_02 = False
    cfg.ACTIONS_ENABLE_ADD_SEQ_DROPOUT_05 = False

    # Act
    model, summary = train_generations(gm, train_loader, val_loader, cfg)

    # Assert
    assert len(summary["generation"]) == 2
    assert GraphStructureQuery.get_amount_of_parameters(model) != params_before
