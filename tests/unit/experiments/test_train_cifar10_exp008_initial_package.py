"""Unit tests for Experiment 008 CIFAR-10 package catalog and frozen Exp 005 settings."""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments import train_cifar10
from experiments.train_cifar10_exp008_initial_package import (
    BATCH_SIZE,
    EPOCHS_20,
    GENERATIONS,
    INITIAL_LR,
    RESIDUAL_CONV_POOL_TYPE,
    RUNS_DIR,
    SCHEDULE_ID,
    SCORE_ACCURACY_METRIC,
    SCORE_WEIGHT_COUNTW,
    SEEDS,
    SIMULATION_ALG,
    SIMULATION_ALG_ID,
    SIMULATION_TIME_SEC,
    SLOPE_ANGLE_THRESHOLD,
    VARIANTS,
    apply_residual_conv_pool_patch,
    hyperparameters_for_variant,
    running_config_for_variant,
    variant_by_id,
)
from experiments.train_mnist_exp004_composed_lr_schedulers import EPOCHS_PER_GENERATION
from growingnn.actions.utils.layer_Factory import ConvFactory
from growingnn.core.config import RunningConfig
import growingnn.core.config as growingnn_config
from growingnn.simulation.simulation_schedulers import (
    AlwaysSimulationScheduler,
    NeverSimulationScheduler,
    SlopeEstimationSimulationScheduler,
)
from growingnn.training.lr_scheduler_global import ComposedLearningRateScheduler
import growingnn.simulation.simulation_algorithms.sequential_halving_beam_alg as sequential_halving_beam_alg


def test_exp008_registers_six_one_factor_variants_and_three_seeds():
    """
    Experiment 008 should compare six CIFAR one-factor variants on seeds 100-102.
    """

    # Arrange / Act
    ids = [variant.variant_id for variant in VARIANTS]

    # Assert
    assert ids == ["base", "narrow", "deep", "epochs20", "always", "fixed"]
    assert SEEDS == (100, 101, 102)


def test_exp008_freezes_experiment_005_package_constants():
    """
    The CIFAR base cell should reuse the finished Exp 005 search, LR, slope, and score package.
    """

    # Arrange / Act / Assert
    assert SIMULATION_ALG_ID == "sequential_halving_beam"
    assert SIMULATION_ALG is sequential_halving_beam_alg
    assert sequential_halving_beam_alg.MAX_DEPTH == 2
    assert sequential_halving_beam_alg.BEAM_WIDTH == 3
    assert SCHEDULE_ID == "composed_exponential"
    assert SLOPE_ANGLE_THRESHOLD == 3.0
    assert GENERATIONS == 10
    assert EPOCHS_PER_GENERATION == 10
    assert SIMULATION_TIME_SEC == 120.0
    assert INITIAL_LR == 0.01
    assert SCORE_ACCURACY_METRIC == "val_acc"
    assert SCORE_WEIGHT_COUNTW == 0.1
    assert BATCH_SIZE == 64
    assert RUNS_DIR.name == "exp008_cifar10_initial_package"


def test_narrow_and_deep_change_starter_capacity_from_base():
    """
    narrow should shrink width and deep should add a second residual block.
    """

    # Arrange
    base = variant_by_id("base")
    narrow = variant_by_id("narrow")
    deep = variant_by_id("deep")

    # Act
    base_model = train_cifar10._build_model(hyperparameters_for_variant(base))
    narrow_model = train_cifar10._build_model(hyperparameters_for_variant(narrow))
    deep_model = train_cifar10._build_model(hyperparameters_for_variant(deep))

    # Assert
    assert (base.channels, base.hidden_dim, base.num_blocks) == (32, 256, 1)
    assert (narrow.channels, narrow.hidden_dim, narrow.num_blocks) == (16, 128, 1)
    assert (deep.channels, deep.hidden_dim, deep.num_blocks) == (32, 256, 2)
    assert base_model.conv1.out_channels == 32
    assert narrow_model.conv1.out_channels == 16
    assert base_model.num_blocks == 1
    assert deep_model.num_blocks == 2
    assert hasattr(deep_model, "layer2")
    probe = torch.randn(2, 3, 32, 32)
    assert base_model(probe).shape == (2, 10)
    assert narrow_model(probe).shape == (2, 10)
    assert deep_model(probe).shape == (2, 10)


def test_epochs20_keeps_base_starter_and_doubles_epochs_per_generation():
    """
    epochs20 should change only the generation length from the base cell.
    """

    # Arrange
    base = variant_by_id("base")
    longer = variant_by_id("epochs20")

    # Act / Assert
    assert longer.channels == base.channels
    assert longer.hidden_dim == base.hidden_dim
    assert longer.num_blocks == base.num_blocks
    assert longer.scheduler == base.scheduler
    assert longer.epochs == EPOCHS_20
    assert longer.epochs == 20
    assert base.epochs == 10


def test_running_config_forces_neuron_resize_flags_off():
    """
    Experiment 008 should keep AddNeurons and DelNeurons off because Exp 006 is unfinished.
    """

    # Arrange
    hp = hyperparameters_for_variant(variant_by_id("base"))

    # Act
    config = running_config_for_variant("slope")(hp, torch.device("cpu"), None)

    # Assert
    assert isinstance(config, RunningConfig)
    assert config.ACTIONS_ENABLE_ADD_NEURONS_11 is False
    assert config.ACTIONS_ENABLE_ADD_NEURONS_15 is False
    assert config.ACTIONS_ENABLE_ADD_NEURONS_20 is False
    assert config.ACTIONS_ENABLE_DEL_NEURONS_01 is False
    assert config.ACTIONS_ENABLE_DEL_NEURONS_05 is False
    assert config.ACTIONS_ENABLE_DEL_NEURONS_09 is False


def test_pooling_patch_sets_residual_to_linear_average_pool():
    """
    The Exp 008 pooling patch should make residual-into-linear skips use AdaptiveAvgPool2d.
    """

    # Arrange / Act
    with apply_residual_conv_pool_patch():
        layer = ConvFactory.create_zero_conv_before_linear(8, 8, 3, 1, 1)

    # Assert
    assert RESIDUAL_CONV_POOL_TYPE == "avg"
    assert growingnn_config.RES_CONV_TO_LINEAR_GLOBAL_POOL_TYPE == "max"
    assert isinstance(layer, nn.Sequential)
    assert isinstance(layer[1], nn.AdaptiveAvgPool2d)


def test_always_and_fixed_select_matching_scheduler_class():
    """
    always should search every generation and fixed should never search.
    """

    # Arrange
    always_hp = hyperparameters_for_variant(variant_by_id("always"))
    fixed_hp = hyperparameters_for_variant(variant_by_id("fixed"))
    base_hp = hyperparameters_for_variant(variant_by_id("base"))

    # Act
    always_cfg = running_config_for_variant(variant_by_id("always").scheduler)(
        always_hp, torch.device("cpu"), None
    )
    fixed_cfg = running_config_for_variant(variant_by_id("fixed").scheduler)(
        fixed_hp, torch.device("cpu"), None
    )
    base_cfg = running_config_for_variant(variant_by_id("base").scheduler)(
        base_hp, torch.device("cpu"), None
    )

    # Assert
    assert isinstance(always_cfg.simulation_scheduler, AlwaysSimulationScheduler)
    assert isinstance(fixed_cfg.simulation_scheduler, NeverSimulationScheduler)
    assert isinstance(base_cfg.simulation_scheduler, SlopeEstimationSimulationScheduler)
    assert base_cfg.simulation_scheduler.angle_threshold == 3.0


def test_hyperparameters_build_composed_exponential_scheduler():
    """
    Each variant should inject the Exp 004 composed_exponential factory.
    """

    # Arrange
    hp = hyperparameters_for_variant(variant_by_id("base"))

    # Act
    scheduler = hp["lr_scheduler_factory"](hp)

    # Assert
    assert callable(hp["lr_scheduler_factory"])
    assert isinstance(scheduler, ComposedLearningRateScheduler)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
