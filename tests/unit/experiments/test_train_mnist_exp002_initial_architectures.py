"""Unit tests for Experiment 002 initial-architecture factories and grid constants."""

import inspect
import re
import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.train_mnist_exp002_initial_architectures import (
    CHANNELS,
    GENERATIONS,
    HIDDEN_LINEAR_SIZE,
    MODEL_VARIANTS,
    RUNS_DIR,
    SEEDS,
    SIMULATION_TIME_SEC,
    SLOPE_ANGLE_THRESHOLD,
    BigAvgPoolMnistNet,
    Medium1Conv2LinearMnistNet,
    Medium2Conv1LinearMnistNet,
    SmallAvgPoolMnistNet,
)

_POOL_ASSIGNMENT = re.compile(
    r"^x = F\.(?:max_pool2d|avg_pool2d|adaptive_max_pool2d|adaptive_avg_pool2d)\b"
)


def test_exp002_uses_four_seeds_and_fixed_three_degree_slope():
    """
    Experiment 002 should use seeds 100-103 and fix the slope at 3°.
    """
    # Arrange / Act / Assert
    assert SEEDS == (100, 101, 102, 103)
    assert SLOPE_ANGLE_THRESHOLD == 3.0


def test_exp002_limits_generations_and_simulation_budget():
    """
    After the first grid, Exp 002 should use fewer generations and a shorter MCTS budget.
    """
    # Arrange / Act / Assert
    assert GENERATIONS == 5
    assert SIMULATION_TIME_SEC == 120.0


def test_exp002_writes_revised_grid_under_after_fix_1_folder():
    """
    Revised Exp 002 runs should land in a separate after_fix_1 output root.
    """
    # Arrange / Act / Assert
    assert RUNS_DIR.name == "exp002_initial_architectures_after_fix_1"


def test_exp002_registers_topology_only_architecture_variants():
    """
    The revised grid should compare layer layout only, not width or pooling style.
    """
    # Arrange / Act
    names = [name for name, _ in MODEL_VARIANTS]

    # Assert
    assert names == [
        "big",
        "medium_1conv_2linear",
        "medium_2conv_1linear",
        "small",
    ]


def test_model_variants_are_ordered_by_descending_start_params():
    """
    MODEL_VARIANTS should list the largest starter first (by trainable parameter count).
    """
    # Arrange / Act
    param_counts = [
        sum(parameter.numel() for parameter in factory({}).parameters())
        for _, factory in MODEL_VARIANTS
    ]

    # Assert
    assert param_counts == sorted(param_counts, reverse=True)
    assert param_counts == [420, 276, 220, 76]


def test_shared_width_constants_match_every_registered_factory():
    """
    Every starter should use the same channel count and the same hidden size when present.
    """
    # Arrange / Act
    models = {name: factory({}) for name, factory in MODEL_VARIANTS}

    # Assert
    assert CHANNELS == 4
    assert HIDDEN_LINEAR_SIZE == 16
    assert models["big"].conv1.out_channels == CHANNELS
    assert models["big"].conv2.out_channels == CHANNELS
    assert models["big"].linear.in_features == CHANNELS
    assert models["big"].linear.out_features == HIDDEN_LINEAR_SIZE
    assert models["medium_1conv_2linear"].conv1.out_channels == CHANNELS
    assert models["medium_1conv_2linear"].linear.in_features == CHANNELS
    assert models["medium_1conv_2linear"].linear.out_features == HIDDEN_LINEAR_SIZE
    assert models["medium_2conv_1linear"].conv1.out_channels == CHANNELS
    assert models["medium_2conv_1linear"].conv2.out_channels == CHANNELS
    assert models["medium_2conv_1linear"].linear2.in_features == CHANNELS
    assert models["small"].conv1.out_channels == CHANNELS
    assert models["small"].linear2.in_features == CHANNELS


def test_no_registered_model_stacks_adjacent_pooling_ops():
    """
    Every Exp 002 starter forward should avoid two pooling ops on consecutive lines.
    """
    # Arrange
    classes = {name: type(factory({})) for name, factory in MODEL_VARIANTS}

    for name, cls in classes.items():
        # Act
        statements = [
            line.strip()
            for line in inspect.getsource(cls.forward).splitlines()
            if line.strip().startswith("x =")
        ]
        pool_flags = [bool(_POOL_ASSIGNMENT.match(statement)) for statement in statements]

        # Assert
        for index in range(len(pool_flags) - 1):
            assert not (pool_flags[index] and pool_flags[index + 1]), name


def test_every_registered_model_uses_only_adaptive_average_pooling():
    """
    Topology comparison should hold pooling fixed to one adaptive_avg_pool2d per starter.
    """
    # Arrange
    classes = {name: type(factory({})) for name, factory in MODEL_VARIANTS}

    for name, cls in classes.items():
        # Act
        source = inspect.getsource(cls.forward)
        pool_lines = [
            line.strip()
            for line in source.splitlines()
            if _POOL_ASSIGNMENT.match(line.strip())
        ]

        # Assert
        assert pool_lines == ["x = F.adaptive_avg_pool2d(x, 1)"], name


def test_each_architecture_factory_forwards_mnist_batch():
    """
    Every registered factory should build a model that accepts (N,1,28,28) and returns (N,10).
    """
    # Arrange
    probe = torch.randn(2, 1, 28, 28)

    for name, factory in MODEL_VARIANTS:
        # Act
        model = factory({})
        output = model(probe)

        # Assert
        assert output.shape == (2, 10), name


def test_single_pool_variants_keep_compact_linear_inputs():
    """
    Adaptive average pool should feed Linear with channel count, not a flattened map.
    """
    # Arrange / Act
    big = BigAvgPoolMnistNet()
    medium = Medium1Conv2LinearMnistNet()
    medium_2c1l = Medium2Conv1LinearMnistNet()
    small = SmallAvgPoolMnistNet()

    # Assert
    assert big.linear.in_features == CHANNELS
    assert medium.linear.in_features == CHANNELS
    assert medium_2c1l.linear2.in_features == CHANNELS
    assert small.linear2.in_features == CHANNELS


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
