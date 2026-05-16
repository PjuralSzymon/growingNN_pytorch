"""Unit tests for ``growingnn.actions.utils.layer_analyser``."""

import torch
import torch.fx as fx

from growingnn.actions.utils.layer_analyser import LayerBridgeFinder, LayerShapeAnalyser
from tests.model_factory import ModelFactory


def test_uniform_activation_shape_returns_shared_shape_when_all_match():
    """
    uniform_activation_shape should return the common tuple when every shape is equal.
    """

    # Arrange / Act
    shape = LayerBridgeFinder.uniform_activation_shape([(1, 8), (1, 8)])

    # Assert
    assert shape == (1, 8)


def test_find_bridge_linear_sizes_maps_rank2_output_to_rank2_input():
    """
    find_bridge_linear_sizes should use the feature dim (last axis) for (batch, features).
    """

    # Arrange / Act
    sizes = LayerBridgeFinder.find_bridge_linear_sizes((1, 8), (1, 16))

    # Assert
    assert sizes == (8, 16)


def test_find_bridge_res_linear_sizes_uses_rank2_outputs():
    """
    find_bridge_res_linear_sizes should read feature dims from both output tensors.
    """

    # Arrange / Act
    sizes = LayerBridgeFinder.find_bridge_res_linear_sizes((2, 4), (2, 8))

    # Assert
    assert sizes == (4, 8)


def test_find_conv_before_linear_sizes_sequential_keeps_channel_width():
    """
    Sequential conv-before-linear should use equal in/out channels on the conv.
    """

    # Arrange / Act
    sizes = LayerBridgeFinder.find_seq_conv_before_linear_sizes((1, 8, 7, 7), (1, 64))

    # Assert
    assert sizes == (8, 8)


def test_find_conv_before_linear_sizes_residual_uses_linear_output_width():
    """
    Residual conv-before-linear should map conv channels to linear output features.
    """

    # Arrange / Act
    sizes = LayerBridgeFinder.find_conv_before_linear_sizes(
        (1, 8, 7, 7), (1, 64), (1, 32), for_residual=True
    )

    # Assert
    assert sizes == (8, 32)


def test_find_equal_conv_output_shapes_requires_matching_rank4():
    """
    find_equal_conv_output_shapes should be true only for equal 4D shapes.
    """

    # Arrange / Act / Assert
    assert LayerBridgeFinder.find_equal_conv_output_shapes((1, 8, 7, 7), (1, 8, 7, 7)) is True
    assert LayerBridgeFinder.find_equal_conv_output_shapes((1, 8, 7, 7), (1, 16, 7, 7)) is False
    assert LayerBridgeFinder.find_equal_conv_output_shapes((1, 8), (1, 8)) is False


def test_find_bridge_linear_sizes_returns_none_for_conv_shapes():
    """
    find_bridge_linear_sizes should reject 4D conv activations, not flatten them.
    """

    # Arrange / Act / Assert
    assert LayerBridgeFinder.find_bridge_linear_sizes((1, 512, 7, 7), (1, 512, 7, 7)) is None


def test_find_bridge_linear_sizes_returns_none_when_shape_missing():
    """
    find_bridge_linear_sizes should return None if either shape is None.
    """

    # Arrange / Act / Assert
    assert LayerBridgeFinder.find_bridge_linear_sizes(None, (1, 4)) is None
    assert LayerBridgeFinder.find_bridge_linear_sizes((1, 4), None) is None


def test_get_layer_output_shapes_records_conv_outputs():
    """
    After ShapeProp, get_layer_output_shapes should list a shape tuple for each
    traced Conv2d submodule (e.g. c1, c2).
    """

    # Arrange
    model = ModelFactory.simple_conv_chain_2()
    gm = fx.symbolic_trace(model)
    x = torch.randn(1, 4, 16, 16)

    # Act
    shapes = LayerShapeAnalyser.get_layer_output_shapes(gm, x)

    # Assert
    assert "c1" in shapes and "c2" in shapes
    assert shapes["c1"] == (1, 4, 16, 16)
    assert shapes["c2"] == (1, 4, 16, 16)


def test_get_layer_input_shapes_matches_predecessor_output():
    """
    get_layer_input_shapes for c2 should match the output shape recorded for c1.
    """

    # Arrange
    model = ModelFactory.simple_conv_chain_2()
    gm = fx.symbolic_trace(model)
    x = torch.randn(1, 4, 16, 16)

    # Act
    out_shapes = LayerShapeAnalyser.get_layer_output_shapes(gm, x)
    in_shapes = LayerShapeAnalyser.get_layer_input_shapes(gm, x)

    # Assert
    assert in_shapes["c2"] == out_shapes["c1"]
    assert in_shapes["c1"] == (1, 4, 16, 16)
