"""Unit tests for delete-layer shape helpers."""

import torch.fx as fx

from growingnn.actions.delete_layer import (
    DelLayer,
    get_common_input_shape,
    get_common_output_shape,
    has_same_input_shape,
    has_same_output_shape,
)
from growingnn.actions.utils.layer_analyser import LayerBridgeFinder, LayerShapeAnalyser
from tests.model_factory import ModelFactory


def test_uniform_activation_shape_rejects_mismatched_rank2_shapes():
    """
    uniform_activation_shape should return None when layer shapes differ.
    """

    # Arrange / Act
    shape = LayerBridgeFinder.uniform_activation_shape([(1, 8), (1, 16)])

    # Assert
    assert shape is None


def test_has_same_output_shape_true_for_matching_probed_shapes():
    """
    has_same_output_shape should use LayerShapeAnalyser output shapes, not nn.Linear fields.
    """

    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_chain_3())
    output_shapes = LayerShapeAnalyser.get_layer_output_shapes(gm)

    # Act
    result = has_same_output_shape(gm, ["l1"], output_shapes=output_shapes)

    # Assert
    assert result is True
    assert get_common_output_shape(gm, ["l1"], output_shapes=output_shapes) == output_shapes["l1"]


def test_has_same_input_shape_false_when_successors_expect_different_shapes():
    """
    has_same_input_shape should be false when probed input shapes for successors differ.
    """

    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_chain_3())

    # Act
    result = has_same_input_shape(
        gm,
        ["l2", "l3"],
        input_shapes={"l2": (1, 8), "l3": (1, 16)},
    )

    # Assert
    assert result is False


def test_del_layer_generate_finds_removable_middle_layer():
    """
    DelLayer.generate_all_actions should propose deleting a middle layer when bridge shapes match.
    """

    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_chain_3())

    # Act
    actions = DelLayer.generate_all_actions(gm)

    # Assert
    assert any(action.params == ["l2"] for action in actions)


def test_get_common_input_shape_returns_none_for_empty_layers():
    """
    get_common_input_shape should return None when no successor layers are given.
    """

    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_chain_3())

    # Act
    shape = get_common_input_shape(gm, [])

    # Assert
    assert shape is None
