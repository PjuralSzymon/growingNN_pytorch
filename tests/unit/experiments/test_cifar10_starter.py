"""Unit tests for CIFAR starter actions and scheduler factories."""

import sys
from pathlib import Path

import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.train_cifar10 import (
    MinimalCifarNet,
    learning_rate_scheduler_factory,
    simulation_scheduler_factory,
)
from growingnn.actions.add_res_conv_layer import AddResConvLayer
from growingnn.actions.add_seq_conv_layer import AddSeqConvLayer
from growingnn.core.config import RunningConfig
from growingnn.core.traced_model import TracedModel
from growingnn.actions.registry import generate_all_actions
from growingnn.simulation.simulation_schedulers import (
    NeverSimulationScheduler,
    SlopeEstimationSimulationScheduler,
)
from growingnn.training.lr_scheduler_global import ComposedLearningRateScheduler


def _hyperparameters(**overrides) -> dict[str, object]:
    values: dict[str, object] = {
        "generations": 1,
        "epochs": 10,
        "lr_alpha": 0.01,
        "simulation_time": 90.0,
        "simulation_epochs": 8,
        "scheduler": "never",
        "lr_mode": "parabolic",
    }
    values.update(overrides)
    return values


def test_zero_residual_on_plain_starter_preserves_outputs():
    """
    AddResConvLayer on the plain CIFAR starter should leave a fixed batch unchanged.
    """
    # Arrange
    torch.manual_seed(0)
    model = MinimalCifarNet(channels=4, hidden_dim=8)
    traced = TracedModel.create(model, (1, 3, 32, 32))
    actions = AddResConvLayer.generate_all_actions(traced)
    assert actions, "plain starter should expose a zero residual convolution"
    inputs = torch.randn(2, 3, 32, 32)
    before = traced.gm(inputs).detach().clone()

    # Act
    actions[0].execute(traced)
    after = traced.gm(inputs).detach()

    # Assert
    assert torch.allclose(before, after, atol=1e-5, rtol=1e-5)


def test_plain_starter_exposes_sequential_convolution():
    """
    The plain CIFAR starter should offer at least one sequential convolution so depth can grow.
    """
    # Arrange
    model = MinimalCifarNet(channels=4, hidden_dim=8)
    traced = TracedModel.create(model, (1, 3, 32, 32))
    config = RunningConfig(generations=1, epochs=1, criterion=nn.CrossEntropyLoss(), quiet=True)

    # Act
    seq_actions = AddSeqConvLayer.generate_all_actions(traced)
    all_actions = generate_all_actions(traced, config)

    # Assert
    assert seq_actions
    assert any(isinstance(action, AddSeqConvLayer) for action in all_actions)


def test_logistic_factory_starts_at_the_target_learning_rate():
    """
    learning_rate_scheduler_factory in logistic mode should already be warmed up before the first action.
    """
    # Arrange
    hp = _hyperparameters(lr_mode="logistic", lr_alpha=0.1, epochs=10)

    # Act
    scheduler = learning_rate_scheduler_factory(hp)
    first = scheduler.alpha_scheduler(0, 10)

    # Assert
    assert first == 0.1


def test_cosine_factory_starts_at_the_global_learning_rate():
    """
    learning_rate_scheduler_factory in cosine mode should start at lr_alpha before any action.
    """
    # Arrange
    hp = _hyperparameters(lr_mode="cosine", lr_alpha=0.01, generations=8, epochs=10)

    # Act
    scheduler = learning_rate_scheduler_factory(hp)
    first = scheduler.alpha_scheduler(0, 10)

    # Assert
    assert isinstance(scheduler, ComposedLearningRateScheduler)
    assert first == 0.01


def test_simulation_scheduler_factory_selects_never_and_slope():
    """
    simulation_scheduler_factory should return never or a 3-degree slope scheduler from the name.
    """
    # Arrange
    never_hp = _hyperparameters(scheduler="never")
    slope_hp = _hyperparameters(scheduler="slope3")

    # Act
    never = simulation_scheduler_factory(never_hp)
    slope = simulation_scheduler_factory(slope_hp)

    # Assert
    assert isinstance(never, NeverSimulationScheduler)
    assert isinstance(slope, SlopeEstimationSimulationScheduler)
    assert slope.angle_threshold == 3.0
