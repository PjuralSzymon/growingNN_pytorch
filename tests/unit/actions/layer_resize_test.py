"""Unit tests for ``growingnn.actions.utils.layer_resize`` propagation."""

import torch
import torch.fx as fx
import torch.nn as nn
import pytest

from growingnn.actions.delete_neurons import DelNeurons
from growingnn.actions.utils.layer_resize import can_resize_linear_output, resize_layer_output
from growingnn.utils.fx import NodeWidthAnalyser, NodeTypeChecker, ModuleResolver
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


def test_shrink_through_relu_updates_fork_consumer_input():
    """
    Shrinking stem behind ReLU should update r1_up.in_features on the forked residual path.
    """

    # Arrange
    gm = fx.symbolic_trace(ModelFactory.complex_residual_many_widths())
    x = torch.randn(2, 4)

    # Act
    resize_layer_output(gm, "stem", 6)
    y = gm(x)

    # Assert
    assert gm.stem.out_features == 6
    assert gm.r1_up.in_features == 6
    assert y.shape == (2, 4)


def test_shrink_through_layer_norm_updates_norm_and_downstream():
    """
    Shrinking merge behind LayerNorm should rescale norm_merge and downstream linears.
    """

    # Arrange
    gm = fx.symbolic_trace(ModelFactory.complex_residual_many_widths_with_activation())
    x = torch.randn(2, 4)

    # Act
    DelNeurons(["merge", 0.5]).execute(gm)
    y = gm(x)

    # Assert
    assert gm.merge.out_features == 5
    assert gm.norm_merge.normalized_shape == (5,)
    assert gm.r4_a.in_features == 5
    assert y.shape == (2, 4)


def test_shrink_stem_on_norm_wrapped_residual_updates_all_add_inputs():
    """
    Shrinking stem on a norm-wrapped residual model must keep the first add inputs aligned.
    """

    # Arrange
    gm = fx.symbolic_trace(ModelFactory.complex_residual_many_widths_with_activation())
    x = torch.randn(2, 4)

    # Act
    DelNeurons(["stem", 0.5]).execute(gm)
    y = gm(x)

    # Assert
    w = gm.stem.out_features
    assert gm.norm_stem.normalized_shape == (w,)
    assert gm.r1_up.in_features == w
    add_node = next(n for n in gm.graph.nodes if NodeTypeChecker.is_add(n))
    widths = [NodeWidthAnalyser.node_output_width(gm, inp) for inp in add_node.all_input_nodes]
    assert len(set(widths)) == 1
    assert y.shape == (2, 4)


def test_mixed_shrink_loop_on_norm_wrapped_residual_stays_consistent():
    """
    A random shrink loop on the norm-wrapped residual model must not break tensor shapes.
    """

    # Arrange
    import random

    gm = fx.symbolic_trace(ModelFactory.complex_residual_many_widths_with_activation())
    x = torch.randn(2, 4)
    rng = random.Random(7)

    # Act
    for _ in range(40):
        actions = DelNeurons.generate_all_actions(gm)
        if not actions:
            break
        DelNeurons(actions[rng.randrange(len(actions))].params).execute(gm)
        gm(x)

    # Assert
    for node in gm.graph.nodes:
        if not NodeTypeChecker.is_add(node):
            continue
        widths = [NodeWidthAnalyser.node_output_width(gm, inp) for inp in node.all_input_nodes]
        assert len(set(widths)) == 1


def test_shrink_hidden_on_conv_residual_fork_resizes_sibling_and_square_seq_linear():
    """
    Shrinking hidden on the CIFAR res_conv fork must sync res_conv, seq_linear_0, and add inputs.
    """

    # Arrange
    gm = fx.symbolic_trace(ModelFactory.cifar_minimal_res_conv_fork_hidden())
    x = torch.randn(2, 3, 32, 32)

    # Act
    DelNeurons(["hidden", 0.9]).execute(gm)
    y = gm(x)

    # Assert
    w = gm.hidden.out_features
    assert gm.seq_linear_0.in_features == w
    assert gm.seq_linear_0.out_features == w
    assert gm.get_submodule("res_conv__0.0").out_channels == w
    assert y.shape == (2, 10)


def test_resize_layer_output_raises_for_non_linear_module():
    """
    resize_layer_output should raise TypeError when the target is not nn.Linear.
    """

    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_conv_chain_2())

    # Act / Assert
    with pytest.raises(TypeError, match="not nn.Linear"):
        resize_layer_output(gm, "c1", 2)
