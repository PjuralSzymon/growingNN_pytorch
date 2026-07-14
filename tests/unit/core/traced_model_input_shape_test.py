"""Unit tests that ShapeProp and actions follow TracedModel.input_shape, not fixed defaults."""

import pytest
import torch.fx as fx

from growingnn.actions.add_seq_conv_layer import AddSeqConvLayer
from growingnn.core.traced_model import TracedModel
from growingnn.utils.fx.graph_analysis import LayerShapeAnalyser
from tests.model_factory import ModelFactory


@pytest.mark.parametrize(
    "input_shape,expected_c1_shape",
    [
        ((1, 4, 16, 16), (1, 4, 16, 16)),
        ((1, 4, 32, 32), (1, 4, 32, 32)),
        ((1, 4, 64, 64), (1, 4, 64, 64)),
    ],
)
def test_traced_model_conv_layer_shapes_match_declared_input(input_shape, expected_c1_shape):
    """
    Conv activation shapes from shapes() should follow TracedModel.input_shape spatial size.
    """
    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_conv_chain_2())
    traced = TracedModel.create(gm, input_shape)

    # Act
    outputs, _ = traced.shapes()

    # Assert
    assert outputs.get("c1") == expected_c1_shape
    assert outputs.get("c2") == expected_c1_shape


@pytest.mark.parametrize(
    "wrong_input_shape",
    [
        (1, 3, 224, 224),
        (1, 8),
        (1, 4),
    ],
)
def test_traced_model_rejects_mismatched_probe_shapes(wrong_input_shape):
    """
    Invalid probe shapes should raise from ShapeProp instead of returning empty layer maps.
    """
    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_conv_chain_2())
    traced = TracedModel.create(gm, wrong_input_shape)

    # Act / Assert
    with pytest.raises(RuntimeError, match="ShapeProp error"):
        traced.shapes()


def test_traced_model_linear_shapes_use_declared_feature_dim():
    """
    Linear layer shapes from shapes() should match the declared (1, features) input.
    """
    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_chain_2())
    traced = TracedModel.create(gm, (1, 4))

    # Act
    outputs, inputs = traced.shapes()

    # Assert
    assert outputs["l1"] == (1, 4)
    assert outputs["l2"] == (1, 4)
    assert inputs["l2"] == (1, 4)


def test_collect_layer_shapes_requires_explicit_probe_source():
    """
    collect_layer_shapes should raise when neither example nor input_shape is provided.
    """
    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_chain_2())

    # Act / Assert
    with pytest.raises(ValueError, match="requires example or input_shape"):
        LayerShapeAnalyser.collect_layer_shapes(gm)


def test_add_seq_conv_actions_use_traced_input_shape_not_image_net_default():
    """
    AddSeqConvLayer should propose actions for dataset-sized probes and raise for 224x224 guesses.
    """
    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_conv_chain_2())
    cifar_traced = TracedModel.create(gm, (1, 4, 32, 32))
    wrong_traced = TracedModel.create(fx.symbolic_trace(ModelFactory.simple_conv_chain_2()), (1, 3, 224, 224))

    # Act
    cifar_actions = AddSeqConvLayer.generate_all_actions(cifar_traced)

    # Assert
    assert len(cifar_actions) >= 1
    with pytest.raises(RuntimeError, match="ShapeProp error"):
        AddSeqConvLayer.generate_all_actions(wrong_traced)
