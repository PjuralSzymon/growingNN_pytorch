"""Unit tests for ``growingnn.actions.utils.layer_resize``."""

import torch.fx as fx

from growingnn.actions.utils.layer_resize import can_resize_linear_output
from tests.model_factory import ModelFactory


def test_can_resize_linear_output_false_for_same_width():
    """
    can_resize_linear_output should return False when the target width equals the current width.
    """

    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_chain_3(neurons=10))

    # Act
    result = can_resize_linear_output(gm, "l2", gm.l2.out_features)

    # Assert
    assert result is False


def test_can_resize_linear_output_false_for_non_linear_module():
    """
    can_resize_linear_output should return False for non-Linear modules.
    """

    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_conv_chain_2())

    # Act
    result = can_resize_linear_output(gm, "c1", 2)

    # Assert
    assert result is False


def test_can_resize_linear_output_true_for_valid_shrink():
    """
    can_resize_linear_output should return True for a shrink that passes matrix and propagation checks.
    """

    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_chain_3(neurons=10))

    # Act
    result = can_resize_linear_output(gm, "l2", 5)

    # Assert
    assert result is True
